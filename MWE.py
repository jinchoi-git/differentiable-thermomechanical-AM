import argparse, os, sys

parser = argparse.ArgumentParser(description="Thermomech runner")
parser.add_argument("mode", nargs="?", default="bfgs",
                    choices=["gradcheck", "adam", "bfgs", "forward", "baseline"],
                    help="Run mode")
parser.add_argument("--iters", type=int, default=10,
                    help="Number of optimizer iterations (Adam/BFGS)")
parser.add_argument("--gpu", type=str, default="0",
                    help="CUDA device id (as string)")
parser.add_argument("--tag", type=str, default=None,
                    help="Optional label appended to the run directory")
parser.add_argument("--ts", action="store_true",
                    help="Add a timestamp subfolder (disabled by default)")

args = parser.parse_args()

# GPU must be set before importing jax
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import time
import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import matplotlib.pyplot as plt
import optax
from collections import namedtuple
from includes.data_loader import load_data
from includes.thermal import ThermContext, simulate_temperature
from includes.mech import MechContext, mech, simulate_mechanics, simulate_mechanics_forward
from includes.utils import save_vtk, find_latest, make_animation_from_pattern, make_iteration_dashboard, save_iter_artifacts, make_run_dir, latest_control_under
import scipy.optimize as spo

# --- Config ---
jax.config.update("jax_enable_x64", False)

learning_rate = 1e-2
work_dir = f"./stress_melt_liq+500_time"
run_dir = make_run_dir(work_dir, args.mode, tag=args.tag, timestamp=args.ts)
print(f"[io] run_dir = {run_dir}")
os.makedirs(run_dir, exist_ok=True)

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
# emissivity = 0.2
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
emissivity = 0.2
solidus = 1878
liquidus = 1928
latent = 286/(liquidus-solidus)
conds = jnp.ones((n_e, 8)) * cond_val
stefan_boltz = 5.670374419e-8  # W·m⁻²·K⁻⁴ (keep units consistent with your nondim if any)

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

tctx = ThermContext(
    # mesh
    n_n=int(n_n), n_e=int(n_e), n_q=int(n_q),
    elements=elements, nodes=nodes, Nip_ele=Nip_ele, Bip_ele=Bip_ele,

    # surfaces (pass the real arrays you use; if unused, keep placeholders or drop fields)
    surfaces=surfaces, Nip_sur=Nip_sur, surf_detJacs=surf_detJacs,
    surface_xy=surface_xy, surface_flux=surface_flux,

    # constants
    ambient=float(ambient), density=float(density),
    cp_val=float(cp_val), cond_val=float(cond_val), conds=conds,

    h_conv=float(h_conv), emissivity=float(emissivity), stefan_boltz=float(stefan_boltz),

    solidus=float(solidus), liquidus=float(liquidus), latent=float(latent),

    base_power=float(base_power), r_beam=float(r_beam),

    laser_loc=laser_loc, 
    laser_on=laser_on,

    element_birth = element_birth,
    node_birth    = node_birth,
    surface_birth = surface_birth,

    dt=float(dt), steps=int(steps),
    BOT_NODES=BOT_NODES,
)

mctx = MechContext(
    n_n=n_n,
    n_e=n_e,
    n_q=n_q,
    dt  = dt,
    steps = steps,
    element_birth = element_birth,
    node_birth    = node_birth,
    surface_birth = surface_birth,
    elements=elements,
    nodes=nodes,
    Nip_ele=Nip_ele,
    Bip_ele=Bip_ele,
    ambient=float(ambient),
    poisson=float(poisson),
    a1=float(a1),
    young1=young1,
    temp_young1=temp_young1,
    Y1=Y1,
    temp_Y1=temp_Y1,
    scl1=scl1,
    temp_scl1=temp_scl1,
    BOT_HEIGHT=float(BOT_HEIGHT),
    Maxit=int(Maxit),
)

# --- Loss + Training ---
def vm_from_S(Sf):
    s11,s22,s33,s12,s23,s13 = (Sf[...,0],Sf[...,1],Sf[...,2],Sf[...,3],Sf[...,4],Sf[...,5])
    vm2 = 0.5*((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2 + 6.0*(s12**2+s23**2+s13**2))
    return jnp.sqrt(jnp.clip(vm2, 0.0, 1e16) + 1e-12)

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
# weights you can tune
STRESS_W = 1.0          # weight on stress loss
MELT_W   = 100.0          # weight on melt loss
SMOOTH_ALPHA = 30.0     # smooth-max sharpness; higher → closer to max

def main_function(params):
    control, _ = expand_params_to_controls(params, power_on_steps, power_off_steps)
    # control = jnp.ones((power_on_steps,), dtype=jnp.float32) * 0.61145806
    # control = jnp.concatenate([control, jnp.zeros((steps - control.shape[0],), control.dtype)], axis=0)

    T = simulate_temperature(control, tctx)   # (steps, n_n)
    S = simulate_mechanics(T, mctx)           # (T_m, n_e, n_q, 6)

    Sf = S[-1]
    s11,s22,s33,s12,s23,s13 = (Sf[...,0],Sf[...,1],Sf[...,2],Sf[...,3],Sf[...,4],Sf[...,5])
    vm2 = 0.5 * ((s11 - s22)**2 + (s22 - s33)**2 + (s33 - s11)**2
                 + 6.0 * (s12**2 + s23**2 + s13**2))
    vm2 = jnp.clip(vm2, 0.0, jnp.inf)   # keep non-negative
    stress_loss = jnp.mean(vm2)

    # smooth max over time for each node: (n_n,)
    #   softmax_max(T) = (1/alpha) * logsumexp(alpha*T)
    Tmax_soft = logsumexp(SMOOTH_ALPHA * T, axis=0) / SMOOTH_ALPHA

    # only penalize build nodes (exclude substrate)
    # build_nodes is (n_n,) mask 0/1 that you already define in the runner
    liquidus = tctx.liquidus
    margin = 500 

    # deficit below liquidus, clamped at 0 if the node reached melt at least once
    deficit = jnp.maximum(liquidus + margin - Tmax_soft, 0.0)

    # normalize by number of build nodes; small eps for safety in x64
    denom = jnp.maximum(jnp.sum(build_nodes), 1.0)
    melt_loss = jnp.sum(deficit * build_nodes) / denom

    # Optionally normalize melt term by liquidus to keep scales similar:
    # melt_loss = melt_loss / (liquidus + 1e-12)

    # ------------------ Total loss ------------------
    total_loss = STRESS_W * stress_loss + MELT_W * melt_loss

    jax.debug.print(
        "loss: total={tot:.4e} | stress={sl:.4e} | melt={ml:.4e} | melt_max_deficit={md:.3f}",
        tot=total_loss, sl=stress_loss, ml=melt_loss, md=jnp.max(deficit)
    )

    return total_loss, control

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

def optimize_adam(params_init, num_iterations, run_dir, learning_rate=1e-3,
                  build_animations=True):
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

        # per-iteration artifacts (pngs + npy)
        save_iter_artifacts(
            iteration=iteration,
            params_np=np.array(params),
            control_np=np.array(control),
            loss_history=loss_history,
            run_dir=run_dir,
            power_on_steps=power_on_steps,
        )

    make_animation_from_pattern(run_dir, "params_plot_*.png",       out_stem="params_anim",  fps=2)
    make_animation_from_pattern(run_dir, "control_plot_*.png",      out_stem="control_anim", fps=2)
    make_animation_from_pattern(run_dir, "loss_history_plot_*.png", out_stem="loss_anim",    fps=2)
    make_iteration_dashboard(run_dir, out_stem="dashboard_anim", fps=2)

    return params, loss_history, control_history

def optimize_bfgs(params_init, num_iterations, run_dir, learning_rate=None):
    os.makedirs(run_dir, exist_ok=True)
    params0 = np.asarray(params_init, dtype=np.float64)

    def loss_only(p_jax):
        return main_function(p_jax)[0]

    loss_and_grad = jax.jit(jax.value_and_grad(loss_only))

    loss_history = []
    control_history = []

    # --- initial snapshot (iteration 0) ---
    p0_jax = jnp.asarray(params0, dtype=jnp.float32)
    L0, control0 = main_function(p0_jax)  # use main_function to get control
    loss_history.append(float(L0))
    control_history.append(np.array(control0))
    save_iter_artifacts(
        iteration=0,
        params_np=np.array(p0_jax),
        control_np=np.array(control0),
        loss_history=loss_history,
        run_dir=run_dir,
        power_on_steps=power_on_steps,
    )

    # cache for callback
    eval_cache = {"last_loss": float(L0), "last_control": np.array(control0), "iter": 1}

    def fun_and_grad(x_np):
        x_jax = jnp.asarray(x_np, dtype=jnp.float32)
        val, grad = loss_and_grad(x_jax)
        control, _ = expand_params_to_controls(x_jax, power_on_steps, power_off_steps)
        eval_cache["last_loss"] = float(val)
        eval_cache["last_control"] = np.asarray(control)
        return float(val), np.asarray(grad, dtype=np.float64)

    def cb(xk):
        it = eval_cache["iter"]
        loss_history.append(eval_cache["last_loss"])
        control_history.append(eval_cache["last_control"])
        save_iter_artifacts(
            iteration=it,
            params_np=np.array(xk, dtype=np.float32),
            control_np=eval_cache["last_control"],
            loss_history=loss_history,
            run_dir=run_dir,
            power_on_steps=power_on_steps,
        )
        print(f"[LBFGS] iter {it:03d}  loss={loss_history[-1]:.6e}")
        eval_cache["iter"] = it + 1

    res = spo.minimize(
        fun_and_grad, x0=params0, method="L-BFGS-B", jac=True,
        options=dict(maxiter=int(num_iterations), gtol=1e-6, ftol=1e-10, maxcor=10, maxls=40),
        callback=cb,
    )

    print(f"[LBFGS] status={res.status}  message={res.message}")
    trained_params = res.x.astype(np.float32)

    # (optional) save consolidated “latest” files
    np.save(os.path.join(run_dir, "params_bfgs_latest.npy"), trained_params)
    np.save(os.path.join(run_dir, "loss_bfgs_latest.npy"), np.array(loss_history))
    
    # After: trained_params, loss_history, control_history = optimize_bfgs(...)
    # Build per-plot animations
    make_animation_from_pattern(run_dir, "params_plot_*.png",         out_stem="params_anim",  fps=2)
    make_animation_from_pattern(run_dir, "control_plot_*.png",        out_stem="control_anim", fps=2)
    make_animation_from_pattern(run_dir, "loss_history_plot_*.png",   out_stem="loss_anim",    fps=2)

    # Optional: single side-by-side dashboard per iteration (params | control | loss)
    make_iteration_dashboard(run_dir, out_stem="dashboard_anim", fps=2)

    return trained_params, np.array(loss_history), np.array(control_history, dtype=object)

if __name__ == "__main__":
    t_start = time.time()
    mode = args.mode
    num_iters = int(args.iters)

    print(f"Using GPU device: {args.gpu}")
    print(f"Mode: {mode} | Iters: {num_iters}")

    if mode == "gradcheck":
        print("Running gradient check...")
        init_params = np.array(params)
        grad_check(init_params, eps=1e-3, n_checks=10)

    elif mode == "adam":
        print("Running optimization (Adam)...")
        trained_params, loss_history, control_history = optimize_adam(
            params_init=params,
            num_iterations=num_iters,           # ← use CLI
            run_dir=run_dir,
            learning_rate=learning_rate,
        )

    elif mode == "bfgs":
        print("Running bfgs optimization...")
        trained_params, loss_history, control_history = optimize_bfgs(
            params_init=params,
            num_iterations=num_iters,           # ← use CLI
            run_dir=run_dir,
        )

    elif mode == "forward":
        print("Running forward simulation...")

        found_any = False
        for src in ["adam", "bfgs"]:
            try:
                ctrl_path, src_run_dir = latest_control_under(work_dir, src)
            except FileNotFoundError as e:
                print(f"[forward] Skip {src}: {e}")
                continue

            found_any = True
            print(f"[forward] Using control from {src}: {ctrl_path}")

            control = jnp.array(np.load(ctrl_path), dtype=jnp.float32)
            if control.shape[0] < steps:
                control = jnp.concatenate(
                    [control, jnp.zeros((steps - control.shape[0],), control.dtype)], axis=0
                )
            elif control.shape[0] > steps:
                control = control[:steps]

            # simulate
            temperatures = simulate_temperature(control, tctx)
            S_seq, U_seq = simulate_mechanics_forward(temperatures, mctx)

            # save next to the source run dir
            out_dir = os.path.join(src_run_dir, "forward")
            os.makedirs(out_dir, exist_ok=True)
            save_vtk(
                temperatures, S_seq, U_seq,
                elements, Bip_ele, nodes, element_birth, node_birth, dt,
                run_dir=out_dir, keyword=f"{src}-forward"
            )
            print(f"[forward] Wrote VTKs to: {out_dir}")

        if not found_any:
            print("[forward] No controls found under run_dir/adam or run_dir/bfgs.")

    elif mode == "baseline":
        print("Running baseline simulation...")
        control = jnp.ones((power_on_steps,), dtype=jnp.float32) * 0.61145806
        control = jnp.concatenate([control, jnp.zeros((steps - control.shape[0],), control.dtype)], axis=0)
        temperatures = simulate_temperature(control, tctx)
        S_seq, U_seq = simulate_mechanics_forward(temperatures, mctx)
        save_vtk(temperatures, S_seq, U_seq, elements, Bip_ele, nodes, element_birth, node_birth, dt,
                 run_dir=run_dir, keyword="baseline")

    t_end = time.time()
    print(f"✅ Total Time: {t_end - t_start:.2f} seconds")