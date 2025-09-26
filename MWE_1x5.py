import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from collections import namedtuple
sys.path.append('./includes')
from data_loader import load_data
from thermal import update_birth, calc_cp, update_mvec_stiffness, update_fluxes
from mech import elastic_stiff_matrix, constitutive_problem, compute_E
from utils import save_vtk, find_latest
from jaxopt import LBFGS

# --- Config ---
gpu_id = "0"  
if len(sys.argv) > 2:
    gpu_id = sys.argv[2]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
# jax.config.update("jax_enable_x64", True)

learning_rate = 1e-2
work_dir = f"./opt_regular_ti64_stresslossonlytrivial_lr{learning_rate}_squareloss_1x5_Sfinal_1"
os.makedirs(work_dir, exist_ok=True)

BOT_HEIGHT = -0.9

# --- Load data ---
dt = 0.01
data_dir = 'preprocessed_1x5'
toolpath_name = 'toolpath_1x5.crs'
(
    elements, nodes, surfaces, node_birth, element_birth, surface_birth,
    surface_xy, surface_flux, toolpath, state, endTime,
    Nip_ele, Bip_ele, Nip_sur, Bip_sur, surf_detJacs
) = load_data(data_dir=data_dir, toolpath_name=toolpath_name, dt=0.01)

def mech(
    temperature,
    active_element_inds,
    active_node_inds,
    U,
    E,
    Ep_prev,
    Hard_prev,
    dU,
    current_time
):
    
    # Masks
    mask_e = active_element_inds                # (n_e,)
    mask_n = active_node_inds                   # (n_n,)
    n_dof = n_n * 3

    # # Interpolate temperature at integration points
    temperature_ip = (
        Nip_ele[:, jnp.newaxis, :] @ temperature[elements][:, jnp.newaxis, :, jnp.newaxis].repeat(8, axis=1)
    )[:, :, 0, 0]
    # tau = 50.0  
    # temperature_ip = temperature_ip - tau * jax.nn.softplus((temperature_ip - 2300.0) / tau)

    # Material properties
    young = jnp.interp(temperature_ip, temp_young1, young1)
    shear = young / (2 * (1 + poisson))
    bulk = young / (3 * (1 - 2 * poisson))
    scl = jnp.interp(temperature_ip, temp_scl1, scl1)
    Y = jnp.interp(temperature_ip, temp_Y1, Y1)
    a = a1 * jnp.ones_like(young)

    # Thermal strain
    alpha_Th = jnp.zeros((n_e, n_q, 6)).at[:, :, 0:3].set(scl[:, :, None].repeat(3, axis=2))
    E_th = (temperature_ip[:, :, None].repeat(6, axis=2) - ambient) * alpha_Th
    E_th = E_th * mask_e[:, None, None]
    
    # Elastic matrices
    ele_K, ele_B, ele_D, ele_detJac = jax.vmap(
        elastic_stiff_matrix, in_axes=(0, None, None, 0, 0)
    )(elements, nodes, Bip_ele, shear, bulk)

    # Zero out inactive elements
    ele_K      = ele_K * mask_e[:, None, None]
    ele_B      = ele_B * mask_e[:, None, None, None]
    ele_D      = ele_D * mask_e[:, None, None, None]
    ele_detJac = ele_detJac * mask_e[:, None]
    B_T = jnp.transpose(ele_B, (0, 1, 3, 2))  # (n_e, 8, 24, 6)
    
    # Dirichlet BC mask at DOF level
    Q_node = jnp.where(nodes[:, 2] < BOT_HEIGHT, 0.0, 1.0) * mask_n   # (n_n,)
    Q_dof  = jnp.repeat(Q_node, 3)                              # (n_dof,)
    elem_dofs = jnp.repeat(elements * 3, 3, axis=1) + jnp.tile(jnp.arange(3), (n_e, 8))

    def newton_iteration(i, U_it):
        E_base = jax.vmap(compute_E, in_axes=(0, 0, None))(elements, ele_B, U_it)
        E_corr = (E_base - E_th) * mask_e[:, None, None]
        S, DS, _, _, _ = constitutive_problem(E_corr, Ep_prev, Hard_prev, shear, bulk, a, Y)

        D_diff    = (ele_detJac[:, :, None, None] * DS) - ele_D
        B_T_D_B   = jnp.sum(B_T @ D_diff @ ele_B, axis=1)
        K_tangent = ele_K + B_T_D_B

        detS   = ele_detJac[..., None] * S
        F_e    = jnp.einsum("eqik,eqk->ei", B_T, detS) * mask_e[:, None]
        F_node = jnp.zeros((n_dof,))
        F_node = F_node.at[elem_dofs.flatten()].add(F_e.flatten())
        
        def assemble_global(K_tangent, elem_dofs, n_dof, mask_e):
            K_global = jnp.zeros((n_dof, n_dof))
            def body(K, e_idx):
                Ke = K_tangent[e_idx] * mask_e[e_idx]
                dofs = elem_dofs[e_idx]
                return K.at[dofs[:, None], dofs[None, :]].add(Ke), None
            K_global, _ = jax.lax.scan(body, K_global, jnp.arange(K_tangent.shape[0]))
            return K_global

        def apply_dirichlet(K_global, resid, Q_dof):
            # Mask out fixed DOFs by multiplying
            K_global = K_global * (Q_dof[:, None] * Q_dof[None, :])
            # Set diagonal to 1 for fixed DOFs
            K_global = K_global + jnp.diag(1.0 - Q_dof)
            # Mask out residuals
            resid = resid * Q_dof
            return K_global, resid

        K_global = assemble_global(K_tangent, elem_dofs, n_dof, mask_e)
        K_global = K_global + 1e-6 * jnp.eye(K_global.shape[0]) # Tikhonov
        resid = -F_node
        K_global, resid = apply_dirichlet(K_global, resid, Q_dof)
        dU_flat = jnp.linalg.solve(K_global, resid)
        dU_new = dU_flat.reshape((n_n, 3))
                
        return U_it + dU_new
            
    U_it = jax.lax.fori_loop(0, Maxit, newton_iteration, U * mask_n[:, None])
    dU = U_it - U

    # Final stress for output
    E_base = jax.vmap(compute_E, in_axes=(0, 0, None))(elements, ele_B, U_it)
    E_corr = (E_base - E_th) * mask_e[:, None, None]
    S_final, DS, IND_p, Ep_new, Hard_new = constitutive_problem(E_corr, Ep_prev, Hard_prev, shear, bulk, a, Y)
    U = jax.lax.dynamic_update_slice(U, U_it, (0, 0))


    return (
        S_final,
        U,
        E_corr,
        Ep_new,
        Hard_new,
        dU,
    )

# --- Thermal simulation ---
def simulate_temperature(control):
    def thermal_step(carry, t_control):
        temperature, temperatures = carry
        t, control_t = t_control
        current_time = t * dt
        ae, an, asurf = update_birth(current_time, element_birth, node_birth, surface_birth)

        cps = jax.vmap(calc_cp, in_axes=(0, None, None, None, None, None, None))(
                    elements, temperature, Nip_ele, solidus, liquidus, cp_val, latent)
        m_vec = jnp.zeros(n_n)
        rhs = jnp.zeros(n_n)
        m_vec, rhs = update_mvec_stiffness(m_vec, rhs, cps, conds, elements, temperature, 
                                           ae, an, nodes, Bip_ele, Nip_ele, density)
        rhs = update_fluxes(t, rhs, temperature, base_power, control_t, asurf,
                            surfaces, surf_detJacs, surface_xy, surface_flux, Nip_sur,
                            nodes, laser_loc, laser_on[t], r_beam, h_conv, ambient, h_rad)

        update = dt * rhs / (m_vec + 1e-8) * an    
        temperature = temperature + update
        temperature = temperature.at[BOT_NODES].set(ambient)
        temperatures = temperatures.at[t].set(temperature)
        return (temperature, temperatures), None

    t_seq = jnp.arange(0, steps)
    control_seq = control
    carry = (
        jnp.full((n_n,), ambient),
        jnp.full((steps, n_n), ambient)
    )
    (temperature_final, temperatures), _ = jax.lax.scan(thermal_step, carry, (t_seq, control_seq))
    return temperatures

# --- Mechanical simulation ---
def simulate_mechanics(temperatures):
    def mech_scan_step(carry, t):
        state = carry
        T_t = temperatures[t]
        current_time = t * dt
        ae, an, _ = update_birth(current_time, element_birth, node_birth, surface_birth)

        S, U, E, Ep_new, Hard_new, dU = mech(
            T_t, ae, an, state.U, state.E, state.Ep_prev, state.Hard_prev,
            state.dU, current_time
        )
        new_state = MechState(U, E, Ep_new, Hard_new, dU)

        return new_state, S

    initial_mech_state = MechState(
        U=jnp.zeros((n_n, 3)),
        E=jnp.zeros((n_e, n_q, 6)),
        Ep_prev=jnp.zeros((n_e, n_q, 6)),
        Hard_prev=jnp.zeros((n_e, n_q, 6)),
        dU=jnp.zeros((n_n, 3)),
    )

    # mech_on = jnp.arange(0, power_on_steps, 10)
    # mech_off = jnp.arange(power_on_steps, steps, 100)
    # mech_timesteps = jnp.concatenate([mech_on, mech_off], axis=0)
    # jax.debug.print("Mechanical steps: {n}", n=mech_timesteps)
    
    mech_timesteps = jnp.arange(0, steps, 10)   
    final_state, S_seq = jax.lax.scan(mech_scan_step, initial_mech_state, mech_timesteps)
    
    return S_seq

def simulate_mechanics_forward(temperatures):
    def mech_scan_step(carry, t):
        state = carry
        T_t = temperatures[t]
        current_time = t * dt
        ae, an, _ = update_birth(current_time, element_birth, node_birth, surface_birth)

        S, U, E, Ep_new, Hard_new, dU = mech(
            T_t, ae, an, state.U, state.E, state.Ep_prev, state.Hard_prev,
            state.dU, current_time
        )
        new_state = MechState(U, E, Ep_new, Hard_new, dU)
        return new_state, (S, new_state.U)

    initial_mech_state = MechState(
        U=jnp.zeros((n_n, 3)),
        E=jnp.zeros((n_e, n_q, 6)),
        Ep_prev=jnp.zeros((n_e, n_q, 6)),
        Hard_prev=jnp.zeros((n_e, n_q, 6)),
        dU=jnp.zeros((n_n, 3)),
    )
    final_state, (S_seq, U_seq) = jax.lax.scan(mech_scan_step, initial_mech_state, mech_timesteps)
    return S_seq, U_seq

# --- Loss + Training ---
def smooth_max_time(T, alpha=10.0, axis=0):
    # T: (time, space)
    return jax.nn.logsumexp(alpha * T, axis=axis) / alpha

def soft_relu(x, beta=10.0):
    # smooth max(x,0)
    return jax.nn.softplus(beta * x) / beta

def melt_loss_fn(
    temperatures,          # (T, S)
    liquidus,              # scalar or (S,)
    active_mask=None,      # None or (S,) in {0,1}
    alpha=10.0,            # sharpness for smooth max
    beta=10.0,             # sharpness for soft ReLU
    margin=0.0,            # require T_max >= liquidus + margin
    huber_delta=None       # None -> pure L2; else Huber for stability
):
    T_max = smooth_max_time(temperatures, alpha=alpha, axis=0)     # (S,)
    # deficit is positive only if we failed to reach the threshold
    deficit = soft_relu((liquidus + margin) - T_max, beta=beta)  # (S,)

    if active_mask is None:
        mask = jnp.ones_like(T_max)
    else:
        mask = active_mask.astype(T_max.dtype)

    denom = jnp.clip(jnp.sum(mask), 1.0)

    if huber_delta is None:
        per_item = deficit**2
    else:
        d = huber_delta
        # Smooth near zero; linear for large misses
        per_item = jnp.where(
            deficit <= d,
            0.5 * deficit**2,
            d * (deficit - 0.5 * d),
        )

    loss = jnp.sum(mask * per_item) / denom
    metrics = {"T_max_smooth": T_max, "melt_deficit": deficit, "melt_loss": loss}
    return loss, metrics

def overheat_loss_fn(temperatures, T_hi=2700.0, alpha=10.0):
    T_max = smooth_max_time(temperatures, alpha=alpha, axis=0)
    over = soft_relu(T_max - T_hi, beta=10.0)
    return jnp.mean(over**2)

def von_mises_from_S(S_frame):  # S_final: (..., 6)
    s11, s22, s33, s12, s23, s13 = (S_frame[...,0], S_frame[...,1], S_frame[...,2],
                                    S_frame[...,3], S_frame[...,4], S_frame[...,5])
    return jnp.sqrt(0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2
                           + 6.0 * (s12**2 + s23**2 + s13**2)))

# --- Blocked control parameterization ---

def expand_params_to_controls(params, power_on_steps, power_off_steps, min_power=0.5, max_power=1.5):
    """
    params50: (50,) unconstrained trainable parameters (we'll clip at usage)
    returns:
      control_full: (steps,) per-step control for thermal (linear interpolation on 'on' window)
      p_clip: (50,) block values AFTER clipping; these align with mech steps at t = 0,10,20,...,490
    """
    # Clip only at usage; keep the actual trainable params unconstrained
    # p_clip = jnp.clip(params50, min_power, max_power)        # (50,)
    p_clip = params ** 2    # (50,)

    # Make sure our blocks divide evenly (should be 10 if power_on_steps=500)
    stride = power_on_steps // N_BLOCKS                       # expected 10
    # Knot positions exactly on mechanics timesteps:
    knots = jnp.arange(0, power_on_steps, stride, dtype=jnp.float32)   # [0,10,20,...,490] length 50
    t = jnp.arange(power_on_steps, dtype=jnp.float32)                    # [0..power_on_steps-1]

    # Linear interpolation for THERMAL so flux changes smoothly between knots
    control_on_interp = jnp.interp(t, knots, p_clip)          # (power_on_steps,)

    # Pad the "off" tail with zeros
    control_full = jnp.concatenate([control_on_interp, jnp.zeros((power_off_steps,))], axis=0)
    return control_full, p_clip

# --------- main loss -----------
def main_function(params):
#     control = jnp.concatenate(
#     [params**2,
#      jnp.zeros((power_off_steps,), dtype=params.dtype)],
#     axis=0,
# )
    control, _ = expand_params_to_controls(params, power_on_steps, power_off_steps)

    T = simulate_temperature(control)       # (T_th, S)
    S = simulate_mechanics(T)               # (T_m, n_e, n_q, 6)

    Sf = S[-1]
    s11,s22,s33,s12,s23,s13 = (Sf[...,0],Sf[...,1],Sf[...,2],Sf[...,3],Sf[...,4],Sf[...,5])
    vm2 = 0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2
                 + 6.0 * (s12**2 + s23**2 + s13**2))
    
    jax.debug.print("max VM^2: {v:.4e}", v=jnp.max(vm2))

    # clip tiny for numerical safety
    vm2 = jnp.clip(vm2, 0.0, 1e16)
    stress_loss = jnp.mean(vm2)

    # stress_loss = jnp.sum(Sf ** 2)

    jax.debug.print("final VM^2: {v:.4e}", v=stress_loss)
    return stress_loss, control

def grad_check(params, eps=1e-3, n_checks=5):
    """
    Compare JAX autodiff gradient of main_function to central finite differences.

    Args:
      params: 1D numpy array of initial control parameters.
      eps:    Finite‐difference step size.
      n_checks: Number of parameters (from index 0) to compare.

    Returns:
      A NumPy array of shape (n_checks, 4) with columns:
        [autodiff_grad, numeric_grad, abs_error, rel_error].
    """
    def loss_fn(p):
        return main_function(jnp.array(p))[0]

    autodiff_grad = np.array(jax.grad(loss_fn)(jnp.array(params)))
    numeric_grad = np.zeros_like(autodiff_grad)
    for i in range(n_checks):
        p_plus  = params.copy();  p_plus[i]  += eps
        p_minus = params.copy();  p_minus[i] -= eps
        f_plus  = float(loss_fn(p_plus))
        f_minus = float(loss_fn(p_minus))
        numeric_grad[i] = (f_plus - f_minus) / (2 * eps)

    table = []
    for i in range(n_checks):
        ag = autodiff_grad[i]
        ng = numeric_grad[i]
        err = abs(ag - ng)
        rel = err / (abs(ng) + 1e-8)
        table.append((i, ag, ng, err, rel))

    print(f"{'idx':>3} │ {'autodiff':>12} │ {'numeric':>12} │ {'abs err':>10} │ {'rel err':>10}")
    print("─────┼" + "─"*14 + "┼" + "─"*14 + "┼" + "─"*12 + "┼" + "─"*12)
    for idx, ag, ng, err, rel in table:
        print(f"{idx:3d} │ {ag:12.6e} │ {ng:12.6e} │ {err:10.2e} │ {rel:10.2e}")

    return np.array(table)

def optimize(params_init, num_iterations, work_dir, learning_rate=1e-3):
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params_init)
    params = params_init

    loss_history = []
    control_history = []

    for iteration in range(num_iterations):
        print(f"\n--- Iteration {iteration} ---")
        start_time = time.time()

        (loss, control), grads = jax.value_and_grad(main_function, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        duration = time.time() - start_time
        print(f"Loss: {loss:.6f} | Time: {duration:.2f}s")

        # Save logs
        loss_history.append(loss)
        control_history.append(control)

        if iteration % 10 == 0:
            # Save parameters
            np.save(os.path.join(work_dir, f"params_{iteration:04d}.npy"), np.array(params))
            np.save(os.path.join(work_dir, f"control_{iteration:04d}.npy"), np.array(control))
            np.save(os.path.join(work_dir, f"loss_{iteration:04d}.npy"), np.array(loss_history))

            # Save plot
            plt.figure(figsize=(8, 4))
            plt.plot(np.array(params), marker='o', linestyle='-')
            plt.title(f'Params at Iteration {iteration}')
            plt.xlabel('Parameter Index')
            plt.ylabel('Parameter Value')
            plt.tight_layout()
            plt.savefig(os.path.join(work_dir, f"params_plot_{iteration:04d}.png"))
            plt.close()

            plt.figure(figsize=(8, 4))
            plt.plot(np.array(control)[:power_on_steps], marker='o', linestyle='-')
            plt.title(f'Control at Iteration {iteration}')
            plt.xlabel('Time Step')
            plt.ylabel('Control Value')
            plt.tight_layout()
            plt.savefig(os.path.join(work_dir, f"control_plot_{iteration:04d}.png"))
            plt.close()

            plt.figure(figsize=(8, 4))
            plt.plot(np.array(loss_history), marker='o', linestyle='-')
            plt.title(f'Loss history')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.tight_layout()
            plt.savefig(os.path.join(work_dir, f"loss_history_plot.png"))
            plt.close()

    return params, loss_history, control_history

# --- Simulation state containers ---
ThermalState = namedtuple("ThermalState", ["temperature", "temperatures"])
MechState = namedtuple("MechState", ["U", "E", "Ep_prev", "Hard_prev", "dU"])

# Time and mesh
steps = int(endTime / dt) + 1
power_on_steps = 100 # jnp.sum(jnp.array(state))
power_off_steps = steps - power_on_steps
print(f"Total time steps: {steps}, Power ON steps: {power_on_steps}, Power OFF steps: {power_off_steps}")
n_n = len(nodes)
n_e = len(elements)
n_p = 8
n_q = 8

# Material & heat transfer properties (SS316L, constant at 300K)
# ambient = 300.0
# dt = 0.01
# density = 0.008
# cp_val = 0.469
# cond_val = 0.0138
# Qin = 300.0 * 0.4 # absortivitiy
# base_power = Qin
# r_beam = 1.12
# h_conv = 0.00005
# h_rad = 0.2
# solidus = 1648
# liquidus = 1673
# latent = 260 / (liquidus - solidus)
# conds = jnp.ones((n_e, 8)) * cond_val

# material properties (TI64)
ambient = 300.0
density = 0.0044
cp_val = 0.714 # at 1073
cond_val = 0.01780 # at 1073
Qin = 400.0 * 0.4
base_power = Qin
r_beam = 1.12
h_conv = 0.00005
h_rad = 0.2
solidus = 1878
liquidus = 1928
latent = 286/(liquidus-solidus)
conds = jnp.ones((n_e, 8)) * cond_val

# Dirichlet boundary
BOT_NODES = nodes[:, 2] < BOT_HEIGHT
build_nodes = (nodes[:, 2] > 0.1).astype(jnp.float32)   # (S,)
N_nodes = jnp.maximum(jnp.sum(build_nodes), 1.0)

# Laser activation
laser_loc = jnp.array(toolpath)
laser_on = jnp.array(state)

# Material models constant
# poisson = 0.3
# a1 = 10000
# young1 = 1.88E5
# Y1 = 2.90E2 * (2/3) ** 0.5
# scl1 = 1.56361E-05

# Material models ti64
poisson = 0.3
a1 = 10000
young1 = jnp.array(np.loadtxt('./materials/TI64_Young_Debroy.txt')[:, 1]) / 1e6
temp_young1 = jnp.array(np.loadtxt('./materials/TI64_Young_Debroy.txt')[:, 0])
Y1 = jnp.array(np.loadtxt('./materials/TI64_Yield_Debroy.txt')[:, 1]) / 1e6 * jnp.sqrt(2/3)
temp_Y1 = jnp.array(np.loadtxt('./materials/TI64_Yield_Debroy.txt')[:, 0])
scl1 = jnp.array(np.loadtxt('./materials/TI64_Alpha_Debroy.txt')[:, 1])
temp_scl1 = jnp.array(np.loadtxt('./materials/TI64_Alpha_Debroy.txt')[:, 0])

# Material models ss316L
# poisson = 0.3
# a1 = 10000
# young1 = jnp.array(np.loadtxt('./materials/SS316L_Young.txt')[:, 1]) / 1e6
# temp_young1 = jnp.array(np.loadtxt('./materials/SS316L_Young.txt')[:, 0])
# Y1 = jnp.array(np.loadtxt('./materials/SS316L_Yield.txt')[:, 1]) / 1e6 * jnp.sqrt(2/3)
# temp_Y1 = jnp.array(np.loadtxt('./materials/SS316L_Yield.txt')[:, 0])
# scl1 = jnp.array(np.loadtxt('./materials/SS316L_Alpha.txt')[:, 1])
# temp_scl1 = jnp.array(np.loadtxt('./materials/SS316L_Alpha.txt')[:, 0])

# Newton and CG tolerances
tol = 1e-4
cg_tol = 1e-4
Maxit = 5

# params = jnp.ones((power_on_steps,))
N_BLOCKS = 10  # 50 params for 500 "laser on" steps (10 steps each)
params = jnp.ones((N_BLOCKS,)) # start at nominal power 1.0
# params = jnp.zeros((N_BLOCKS,))  # start at nominal power 0

# def loss_only(p): return main_function(p)[0]

# p = params_init
# g = jax.grad(loss_only)(p)
# jax.debug.print("||g||2={:.3e}  max|g|={:.3e}", jnp.linalg.norm(g), jnp.max(jnp.abs(g)))
# jax.debug.print("g·p={:.3e}", jnp.vdot(g, p))   # should be >0 so stepping -g reduces amplitude

# # finite-diff along +g should be positive
# def fd_dir(p, d, eps=1e-3):
#     d = d / (jnp.linalg.norm(d) + 1e-12)
#     return (loss_only(p + eps*d) - loss_only(p - eps*d)) / (2*eps)
# jax.debug.print("FD(+g)={:.3e}", fd_dir(p, g))

# os.exit()

def determinism_check(p):
    L1 = loss_only(p)
    L2 = loss_only(p)
    jax.debug.print("replay ΔL = {d:.3e}", d=(L2 - L1))

def _fp(x):
    return (jnp.sum(x), jnp.sum(x * x))

def loss_only(p):  # use your existing main_function
    return main_function(p)[0]

def audit_once(p):
    # control, _ = expand_params_to_controls(p, power_on_steps, power_off_steps)

    control = jnp.concatenate(
    [jax.nn.sigmoid(params),
     jnp.zeros((power_off_steps,), dtype=params.dtype)],
    axis=0,
)

    T1 = simulate_temperature(control)
    T2 = simulate_temperature(control)
    jax.debug.print("ΔT sums: Δsum={:.3e}, Δsumsq={:.3e}",
                    _fp(T2)[0]-_fp(T1)[0], _fp(T2)[1]-_fp(T1)[1])

    # If T is stable, check mechanics on fixed T
    S1 = simulate_mechanics(T1)
    S2 = simulate_mechanics(T1)  # reuse same T!
    vm2 = lambda Sf: 0.5*((Sf[...,0]-Sf[...,1])**2+(Sf[...,1]-Sf[...,2])**2+(Sf[...,2]-Sf[...,0])**2
                          + 6*(Sf[...,3]**2+Sf[...,4]**2+Sf[...,5]**2))
    jax.debug.print("ΔS sums: Δsum={:.3e}, Δsumsq={:.3e}",
                    _fp(S2)[0]-_fp(S1)[0], _fp(S2)[1]-_fp(S1)[1])
    jax.debug.print("ΔVM^2(final)={:.3e}",
                    jnp.mean(vm2(S2[-1])) - jnp.mean(vm2(S1[-1])))

# determinism_check(params)
# audit_once(params)
# os.exit()

if __name__ == "__main__":   
    t_start = time.time()

    # Check command-line arguments
    mode = "optimize"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    print(f"Using GPU device: {gpu_id}")

    if mode == "gradcheck":
        print("Running gradient check...")
        init_params = np.array(params) 
        grad_check(init_params, eps=1e-3, n_checks=50)

    elif mode == "optimize":
        print("Running optimization...")
        trained_params, loss_history, control_history = optimize(
            params_init=params,
            num_iterations=100,
            work_dir=work_dir,
            learning_rate=learning_rate,
        )

    elif mode == "bfgs":
        print("Running bfgs optimization...")
        trained_params, loss_history, control_history = optimize(
            params_init=params,
            num_iterations=100,
            work_dir=work_dir,
            learning_rate=learning_rate,
        )

    elif mode == "forward":
        print("Running forward simulation...")        
        control = jnp.load(find_latest("control", work_dir)[0])
        # control = jnp.concatenate((control, jnp.zeros((steps - len(control),))), axis=0)
        temperatures = simulate_temperature(control)
        S_seq, U_seq = simulate_mechanics_forward(temperatures)
        save_vtk(temperatures, S_seq, U_seq, elements, Bip_ele, nodes, element_birth, node_birth, dt, work_dir=work_dir, keyword="forward")

    elif mode == "baseline":
        print("Running baseline simulation...")        
        control = jnp.ones(power_on_steps,) * 1.0
        control = jnp.concatenate((control, jnp.zeros((steps - len(control),))), axis=0)
        temperatures = simulate_temperature(control)
        S_seq, U_seq = simulate_mechanics_forward(temperatures)
        save_vtk(temperatures, S_seq, U_seq, elements, Bip_ele, nodes, element_birth, node_birth, dt, work_dir=work_dir, keyword="baseline")

    else:
        print(f"Unknown mode '{mode}'. Use 'gradcheck' or 'optimize' or 'forward'.")

    t_end = time.time()
    print(f"✅ Total Time: {t_end - t_start:.2f} seconds")