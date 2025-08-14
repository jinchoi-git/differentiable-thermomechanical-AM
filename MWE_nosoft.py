import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import lax
import optax
from collections import namedtuple
import functools
np.bool = np.bool_

sys.path.append('./includes_nosoft')
from data_loader import load_toolpath, shape_fnc_element, derivate_shape_fnc_element, shape_fnc_surface, derivate_shape_fnc_surface, surface_jacobian
from thermal import update_birth, calc_cp, update_mvec_stiffness, update_fluxes
from mech import elastic_stiff_matrix, constitutive_problem, compute_E

# --- Config ---
gpu_id = "0"  # Default GPU
if len(sys.argv) > 2:
    gpu_id = sys.argv[2]
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

output_dir = "./check_gradient_MWE"
os.makedirs(output_dir, exist_ok=True)

jax.config.update("jax_enable_x64", True)

# --- Load data ---
Load_data = True # always True I think
dt = 0.01

data_dir = 'preprocessed_10x5'
toolpath_name = '10x5_toolpath.crs'

if Load_data:
    elements = jnp.load(f'{data_dir}/elements.npy')
    # elements = jnp.load(f'{data_dir}/elements.npy')[0:1]
    nodes = jnp.load(f'{data_dir}/nodes.npy')
    surfaces = jnp.load(f'{data_dir}/surface.npy')
    node_birth = jnp.load(f'{data_dir}/node_birth.npy')
    element_birth = jnp.load(f'{data_dir}/element_birth.npy')
    surface_birth = jnp.load(f'{data_dir}/surface_birth.npy')
    surface_xy = jnp.load(f'{data_dir}/surface_xy.npy')
    surface_flux = jnp.load(f'{data_dir}/surface_flux.npy')
    
toolpath, state, endTime = load_toolpath(filename=toolpath_name, dt = dt)
parCoords_element = jnp.array([[-1.0,-1.0,-1.0],[1.0,-1.0,-1.0],[1.0, 1.0,-1.0],[-1.0, 1.0,-1.0],
             [-1.0,-1.0,1.0],[1.0,-1.0, 1.0], [ 1.0,1.0,1.0],[-1.0, 1.0,1.0]]) * 0.5773502692
parCoords_surface = jnp.array([[-1.0,-1.0],[-1.0, 1.0],[1.0,-1.0],[1.0,1.0]])* 0.5773502692

# Nip_ele = np.array([shape_fnc_element(parCoord) for parCoord in parCoords_element]) #[:,:,jnp.newaxis]
# Bip_ele = np.array([derivate_shape_fnc_element(parCoord) for parCoord in parCoords_element])
# Nip_sur = np.array([shape_fnc_surface(parCoord) for parCoord in parCoords_surface])
# Bip_sur = np.array([derivate_shape_fnc_surface(parCoord) for parCoord in parCoords_surface])
Nip_ele = jnp.stack([shape_fnc_element(pc) for pc in parCoords_element], axis=0)
Bip_ele = jnp.stack([derivate_shape_fnc_element(pc) for pc in parCoords_element], axis=0)
Nip_sur = jnp.stack([shape_fnc_surface(pc) for pc in parCoords_surface], axis=0)
Bip_sur = jnp.stack([derivate_shape_fnc_surface(pc) for pc in parCoords_surface], axis=0)
surf_detJacs = surface_jacobian(nodes, surfaces, Bip_sur)

print("Number of nodes: {}".format(len(nodes)))
print("Number of elements: {}".format(len(elements)))
print("Number of surfaces: {}".format(len(surfaces)))
print("Number of time-steps: {}".format(len(toolpath)))

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
    # tau = 20.0  # smaller = sharper clamp
    # temperature_ip = temperature_ip - tau * jax.nn.softplus((temperature_ip - 2300.0) / tau)
    temperature_ip = jnp.clip(temperature_ip, 300, 2300)

    # Material properties (constants)
    young = jnp.full_like(temperature_ip, young1)
    shear = young / (2 * (1 + poisson))
    bulk = young / (3 * (1 - 2 * poisson))
    scl = jnp.full_like(temperature_ip, scl1)
    Y = jnp.full_like(temperature_ip, Y1)
    a = a1 * jnp.ones_like(young)

    # Thermal strain
    alpha_Th = jnp.zeros((n_e, n_q, 6)).at[:, :, 0:3].set(scl[:, :, None].repeat(3, axis=2))
    E_th = (temperature_ip[:, :, None].repeat(6, axis=2) - T_Ref) * alpha_Th
    E_th = E_th * mask_e[:, None, None]
    
    # Elastic matrices (computed for all elements)
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
    Q_node = jnp.where(nodes[:, 2] < -2.9, 0.0, 1.0) * mask_n   # (n_n,)
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
    
    # Update global U
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

        cps = jax.vmap(
            calc_cp,
            in_axes=(0, None, None, None, None, None, None)
        )(
            elements,         # (n_e,)
            temperature,      # (n_n,) or (n_e, n_n) depending on your model
            Nip_ele,          # (n_q, n_p)
            solidus,          # scalar
            liquidus,         # scalar
            cp_val,           # scalar
            latent            # scalar
        )
        m_vec = jnp.zeros(n_n)
        rhs = jnp.zeros(n_n)
        m_vec, rhs = update_mvec_stiffness(m_vec, rhs, cps, conds, elements, temperature, ae, an, nodes, Bip_ele, Nip_ele, density)
        rhs = update_fluxes(t, rhs, temperature, base_power, control_t, asurf,
                  surfaces, surf_detJacs, surface_xy, surface_flux, Nip_sur, nodes, laser_loc, laser_on, r_beam, h_conv, ambient, h_rad)

        update = dt * rhs / (m_vec + 1e-8) * an    
        
        temperature = temperature + update
        temperature = temperature.at[bot_nodes].set(ambient)
        temperatures = temperatures.at[t].set(temperature)
                       
        return (temperature, temperatures), None

    t_seq = jnp.arange(1, steps)
    control_seq = control[:-1]
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

    mech_timesteps = jnp.arange(1, steps, 10)
    final_state, S_seq = jax.lax.scan(mech_scan_step, initial_mech_state, mech_timesteps)
    
    return S_seq

# --- Loss + Training ---
# def smooth_max_time(T, alpha=20.0, axis=0):
#     # T: (time, space)
#     return jax.nn.logsumexp(alpha * T, axis=axis) / alpha

# def soft_relu(x, beta=20.0):
#     # smooth max(x,0)
#     return jax.nn.softplus(beta * x) / beta

# def melt_loss_fn(
#     temperatures,          # (T, S)
#     liquidus,              # scalar or (S,)
#     active_mask=None,      # None or (S,) in {0,1}
#     alpha=20.0,            # sharpness for smooth max
#     beta=20.0,             # sharpness for soft ReLU
#     margin=0.0,            # require T_max >= liquidus + margin
#     huber_delta=None       # None -> pure L2; else Huber for stability
# ):
#     T_max = smooth_max_time(temperatures, alpha=alpha, axis=0)     # (S,)
#     # deficit is positive only if we failed to reach the threshold
#     deficit = soft_relu((liquidus + margin) - T_max, beta=beta)  # (S,)

#     if active_mask is None:
#         mask = jnp.ones_like(T_max)
#     else:
#         mask = active_mask.astype(T_max.dtype)

#     denom = jnp.clip(jnp.sum(mask), 1.0)

#     if huber_delta is None:
#         per_item = deficit**2
#     else:
#         d = huber_delta
#         # Smooth near zero; linear for large misses
#         per_item = jnp.where(
#             deficit <= d,
#             0.5 * deficit**2,
#             d * (deficit - 0.5 * d),
#         )

#     loss = jnp.sum(mask * per_item) / denom
#     metrics = {"T_max_smooth": T_max, "melt_deficit": deficit, "melt_loss": loss}
#     return loss, metrics

# def main_function(params):
#     control_soft = 0.75 + 0.5 * jax.nn.sigmoid(params[:power_on_steps])
#     control = jnp.concatenate((control_soft, jnp.zeros((steps - power_on_steps,))), axis=0)

#     # simulate_temperature_jit = jax.jit(simulate_temperature)
#     # simulate_mechanics_jit  = jax.jit(simulate_mechanics)
#     # temperatures = simulate_temperature_jit(control)
#     # S_seq = simulate_mechanics_jit(temperatures)
#     temperatures = simulate_temperature(control)
#     S_seq = simulate_mechanics(temperatures)

#     # Compute final von Mises stress
#     S_final = S_seq[-1]  # shape: (n_e, n_q, 6)
    
#     # Von Mises stress (same as in save_vtk)
#     von_mises = jnp.sqrt(0.5 * (
#         (S_final[..., 0] - S_final[..., 1])**2 +
#         (S_final[..., 1] - S_final[..., 2])**2 +
#         (S_final[..., 2] - S_final[..., 0])**2 +
#         6 * (S_final[..., 3]**2 + S_final[..., 4]**2 + S_final[..., 5]**2)
#     ))
#     stress_loss = jnp.mean(von_mises)
    
#     # Ensure all active nodes reached liquidus temperature at some point
#     is_build = (nodes[:, 2] > 0.01).astype(jnp.float64)   # or z > 0 if that matches births
#     melt_loss, _ = melt_loss_fn(temperatures, liquidus, active_mask=is_build, alpha=20.0, beta=20.0, margin=0, huber_delta=None)
    
#     # Final total loss
#     loss = stress_loss #+ melt_loss
    
#     return loss, control

# --- Loss + Training ---
def main_function(params, smooth_weight=1e-2):
    control = jnp.concatenate([
        jnp.clip(params[:power_on_steps], 0.5, 2.0),
        jnp.zeros((steps - power_on_steps,))
    ], axis=0)
    
    smooth_penalty = smooth_weight * jnp.sum((params[1:] - params[:-1])**2)
    temperatures = simulate_temperature(control)
    S_seq = simulate_mechanics(temperatures)

    # Compute final von Mises stress
    S_final = S_seq[-1]  # shape: (n_e, n_q, 6)
    
    # Von Mises stress (same as in save_vtk)
    von_mises = jnp.sqrt(0.5 * (
        (S_final[..., 0] - S_final[..., 1])**2 +
        (S_final[..., 1] - S_final[..., 2])**2 +
        (S_final[..., 2] - S_final[..., 0])**2 +
        6 * (S_final[..., 3]**2 + S_final[..., 4]**2 + S_final[..., 5]**2)
    ))
    stress_loss = jnp.mean(von_mises)
    
    # Ensure all active nodes reached liquidus temperature at some point
    T_max = jnp.max(temperatures, axis=0)  # shape: (n_nodes,)
    T_deficit = jnp.clip(liquidus - T_max, a_min=0.0)
    is_build = nodes[:, 2] > 0.0  # shape (n_nodes,)
    melt_penalty = jnp.sum((T_deficit * is_build) ** 2) / jnp.sum(is_build)
    
    # Final total loss
    loss = stress_loss + 10.0 * melt_penalty + smooth_penalty
    
    return loss , control

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
    # Define a pure‐Python loss function wrapper
    def loss_fn(p):
        # main_function returns (loss, control); we differentiate w.r.t. loss only
        return main_function(jnp.array(p))[0]

    # Compute autodiff gradient via JAX
    autodiff_grad = np.array(jax.grad(loss_fn)(jnp.array(params)))

    # Compute numeric gradient via central finite difference
    numeric_grad = np.zeros_like(autodiff_grad)
    for i in range(n_checks):
        p_plus  = params.copy();  p_plus[i]  += eps
        p_minus = params.copy();  p_minus[i] -= eps
        f_plus  = float(loss_fn(p_plus))
        f_minus = float(loss_fn(p_minus))
        numeric_grad[i] = (f_plus - f_minus) / (2 * eps)

    # Build comparison table
    table = []
    for i in range(n_checks):
        ag = autodiff_grad[i]
        ng = numeric_grad[i]
        err = abs(ag - ng)
        rel = err / (abs(ng) + 1e-8)
        table.append((i, ag, ng, err, rel))

    # Print results
    print(f"{'idx':>3} │ {'autodiff':>12} │ {'numeric':>12} │ {'abs err':>10} │ {'rel err':>10}")
    print("─────┼" + "─"*14 + "┼" + "─"*14 + "┼" + "─"*12 + "┼" + "─"*12)
    for idx, ag, ng, err, rel in table:
        print(f"{idx:3d} │ {ag:12.6e} │ {ng:12.6e} │ {err:10.2e} │ {rel:10.2e}")

    return np.array(table)

def optimize(params_init, num_iterations, output_dir, learning_rate=1e-3, smooth_weight=1e-2):
    # --- Optimizer ---
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params_init)
    params = params_init

    loss_history = []
    control_history = []

    os.makedirs(output_dir, exist_ok=True)

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

        if iteration % 1 == 0:
            # Save parameters
            np.save(os.path.join(output_dir, f"params_{iteration:04d}.npy"), np.array(params))
            np.save(os.path.join(output_dir, f"control_{iteration:04d}.npy"), np.array(control))
            np.save(os.path.join(output_dir, f"loss_{iteration:04d}.npy"), np.array(loss_history))

    return params, loss_history, control_history

# --- Simulation state containers ---
ThermalState = namedtuple("ThermalState", ["temperature", "temperatures"])
MechState = namedtuple("MechState", ["U", "E", "Ep_prev", "Hard_prev", "dU"])

# Time and mesh
steps = int(endTime / dt)
power_on_steps = 500 # jnp.sum(jnp.array(state))
n_n = len(nodes)
n_e = len(elements)
n_p = 8
n_q = 8

# Material & heat transfer properties (SS316L, constant at 300K)
ambient = 300.0
dt = 0.01
density = 0.008
cp_val = 0.469
cond_val = 0.0138
Qin = 400.0 * 0.4
base_power = Qin
r_beam = 1.12
h_conv = 0.00005
h_rad = 0.2
solidus = 1648
liquidus = 1673
latent = 260 / (liquidus - solidus)
conds = jnp.ones((n_e, 8)) * cond_val

# Dirichlet boundary
bot_nodes = nodes[:, 2] < -2.9

# Laser activation
laser_loc = jnp.array(toolpath)
laser_on = jnp.array(state)
element_birth = jnp.array(element_birth)
node_birth = jnp.array(node_birth)

# Material models
poisson = 0.3
a1 = 10000
young1 = 1.88E5
Y1 = 2.90E2 * (2/3) ** 0.5
scl1 = 1.56361E-05
T_Ref = ambient

# Newton and CG tolerances
tol = 1e-4
cg_tol = 1e-4
Maxit = 1

params_init = jnp.ones((power_on_steps,))

if __name__ == "__main__":   
    t_start = time.time()

    # Check command-line arguments
    mode = "optimize"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    print(f"Using GPU device: {gpu_id}")

    if mode == "gradcheck":
        print("Running gradient check...")
        init_params = np.array(params_init) 
        grad_check(init_params, eps=1e-3, n_checks=2)
    elif mode == "optimize":
        print("Running optimization...")
        trained_params, loss_history, control_history = optimize(
            params_init=params_init,
            num_iterations=5,
            output_dir=output_dir,
            learning_rate=1e-2,
            smooth_weight=1e-2
        )
    else:
        print(f"Unknown mode '{mode}'. Use 'gradcheck' or 'optimize'.")

    t_end = time.time()
    print(f"✅ Total Time: {t_end - t_start:.2f} seconds")