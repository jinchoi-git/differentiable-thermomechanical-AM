import argparse, os, sys, json
from pathlib import Path

parser = argparse.ArgumentParser(description="Thermomech runner")
parser.add_argument(
    "mode",
    nargs="?",
    default="bfgs",
    choices=["gradcheck", "adam", "bfgs", "forward", "baseline", "baseline_avg"],
    help="Run mode",
)
parser.add_argument(
    "--iters", type=int, default=10, help="Number of optimizer iterations (Adam/BFGS)"
)
parser.add_argument("--gpu", type=str, default="0", help="CUDA device id (as string)")
parser.add_argument(
    "--tag", type=str, default=None, help="Optional label appended to the run directory"
)
parser.add_argument(
    "--ts", action="store_true", help="Add a timestamp subfolder (disabled by default)"
)

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
from includes.mech import (
    MechContext,
    mech,
    simulate_mechanics,
    simulate_mechanics_forward,
)
from includes.utils import (
    save_vtk,
    find_latest,
    make_animation_from_pattern,
    make_iteration_dashboard,
    save_iter_artifacts,
    make_run_dir,
    latest_control_under,
    get_project_root,
)
import scipy.optimize as spo
from dataclasses import replace

# --- Config ---
jax.config.update("jax_enable_x64", False)

PROJECT_ROOT = get_project_root()
DATA_ROOT = PROJECT_ROOT / "preprocessed"
RESULTS_ROOT = PROJECT_ROOT / "results"

# base_name = '1_stsl'
# base_name = '2_stml'
base_name = "3_mtsl"
# base_name = '4_mtml'
# base_name = '4_mtml_half'
learning_rate = 1e-2
work_dir = RESULTS_ROOT / f"{base_name}_bfgsbound_11params_fix"
run_dir = make_run_dir(str(work_dir), args.mode, tag=args.tag, timestamp=args.ts)
print(f"[io] run_dir = {run_dir}")
os.makedirs(run_dir, exist_ok=True)

BOT_HEIGHT = -0.9

# --- Load data ---
dt = 0.01
data_dir = DATA_ROOT / f"{base_name}_preprocessed"
toolpath_name = DATA_ROOT / f"{base_name}.crs"

(
    elements,
    nodes,
    surfaces,
    node_birth,
    element_birth,
    surface_birth,
    surface_xy,
    surface_flux,
    toolpath,
    state,
    endTime,
    Nip_ele,
    Bip_ele,
    Nip_sur,
    Bip_sur,
    surf_detJacs,
) = load_data(data_dir=data_dir, toolpath_name=toolpath_name, dt=dt)

# Time and mesh
power_on_time = float(toolpath_name.read_text().strip().splitlines()[-2].split()[0])
print(f"endTime from toolpath file: {endTime}")
print(f"power_on_time from toolpath file: {power_on_time}")

steps = int(endTime / dt) + 1
power_on_steps = int(power_on_time / dt)
power_off_steps = steps - power_on_steps
print(
    f"Total time steps: {steps}, Power ON steps: {power_on_steps}, Power OFF steps: {power_off_steps}"
)
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
cp_val = 0.714  # at 1073 0.546 # at 298
cond_val = 0.01780  # at 1073 0.007 # at 298
Qin = 500 * 0.4  # 500 * 0.4 or 350 * 0.4 for half layers
base_power = Qin
r_beam = 1.12
h_conv = 5e-5  # 5e-5
emissivity = 0.2
solidus = 1878
liquidus = 1928
latent = 286 / (liquidus - solidus)
conds = jnp.ones((n_e, 8)) * cond_val
stefan_boltz = 5.670374419e-8  # W*m^-2*K^-4

# Dirichlet boundary
BOT_NODES = nodes[:, 2] < BOT_HEIGHT
build_nodes = (nodes[:, 2] > 0.1).astype(jnp.float32)  # (S,)

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
young1 = jnp.array(np.loadtxt("./materials/TI64_Young_Debroy.txt")[:, 1]) / 1e6
temp_young1 = jnp.array(np.loadtxt("./materials/TI64_Young_Debroy.txt")[:, 0])
Y1 = (
    jnp.array(np.loadtxt("./materials/TI64_Yield_Debroy.txt")[:, 1])
    / 1e6
    * jnp.sqrt(2 / 3)
)
temp_Y1 = jnp.array(np.loadtxt("./materials/TI64_Yield_Debroy.txt")[:, 0])
scl1 = jnp.array(np.loadtxt("./materials/TI64_Alpha_Debroy.txt")[:, 1])
temp_scl1 = jnp.array(np.loadtxt("./materials/TI64_Alpha_Debroy.txt")[:, 0])

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
Maxit = 8

# params = jnp.ones((power_on_steps,))
N_BLOCKS = (
    11  # power_on_steps // 10 # 50 params for 500 "laser on" steps (10 steps each)
)
params = jnp.ones((N_BLOCKS,))  # start at nominal power 1.0

tctx = ThermContext(
    # mesh
    n_n=int(n_n),
    n_e=int(n_e),
    n_q=int(n_q),
    elements=elements,
    nodes=nodes,
    Nip_ele=Nip_ele,
    Bip_ele=Bip_ele,
    # surfaces (pass the real arrays you use; if unused, keep placeholders or drop fields)
    surfaces=surfaces,
    Nip_sur=Nip_sur,
    surf_detJacs=surf_detJacs,
    surface_xy=surface_xy,
    surface_flux=surface_flux,
    # constants
    ambient=float(ambient),
    density=float(density),
    cp_val=float(cp_val),
    cond_val=float(cond_val),
    conds=conds,
    h_conv=float(h_conv),
    emissivity=float(emissivity),
    stefan_boltz=float(stefan_boltz),
    solidus=float(solidus),
    liquidus=float(liquidus),
    latent=float(latent),
    base_power=float(base_power),
    r_beam=float(r_beam),
    laser_loc=laser_loc,
    laser_on=laser_on,
    element_birth=element_birth,
    node_birth=node_birth,
    surface_birth=surface_birth,
    dt=float(dt),
    steps=int(steps),
    BOT_NODES=BOT_NODES,
)

mctx = MechContext(
    n_n=n_n,
    n_e=n_e,
    n_q=n_q,
    dt=dt,
    steps=steps,
    element_birth=element_birth,
    node_birth=node_birth,
    surface_birth=surface_birth,
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

# choose ranges for control during "on" window
CTRL_MIN = 0
CTRL_MAX = 1.5  # or whatever makes sense


def expand_blocks_to_controls(block_controls, power_on_steps, power_off_steps):
    """
    block_controls: (N_BLOCKS,) already in [CTRL_MIN, CTRL_MAX] due to L-BFGS-B bounds
    Returns:
      control_full: (steps,) per-step control for thermal (interp on 'on' window)
    """
    stride = power_on_steps // N_BLOCKS
    knots = jnp.arange(0, power_on_steps, stride)
    t = jnp.arange(power_on_steps)
    control_on_interp = jnp.interp(t, knots, block_controls)  # smooth over on-window
    control_full = jnp.concatenate(
        [control_on_interp, jnp.zeros((power_off_steps,))], axis=0
    )
    return control_full


def constant_on_from_control(control, power_on_steps, power_off_steps):
    """Return a baseline control: constant = mean(control[:power_on_steps]) during ON, 0 during OFF."""
    control = jnp.asarray(control, dtype=jnp.float32)
    on = control[:power_on_steps]
    const_val = jnp.mean(on) if on.size > 0 else jnp.array(0.0, dtype=jnp.float32)
    control_on = jnp.full((power_on_steps,), const_val, dtype=jnp.float32)
    control_off = jnp.zeros((power_off_steps,), dtype=jnp.float32)
    return jnp.concatenate([control_on, control_off], axis=0), const_val


def von_mises_from_S(S_last):
    """
    S_last: (..., 6) in Voigt order [s11, s22, s33, s12, s23, s13]
    returns vm: (...) pointwise von Mises
    """
    s11, s22, s33, s12, s23, s13 = (
        S_last[..., 0],
        S_last[..., 1],
        S_last[..., 2],
        S_last[..., 3],
        S_last[..., 4],
        S_last[..., 5],
    )
    vm2 = 0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3.0 * (
        s12**2 + s23**2 + s13**2
    )
    return jnp.sqrt(jnp.clip(vm2, 0.0, jnp.inf))


# --- Loss helpers (pure; no side-effects) ---
def stress_loss_from_S_final(S_final):
    """Mean of von Mises^2 at the final mechanics output (matches optimization)."""
    s11, s22, s33, s12, s23, s13 = (
        S_final[..., 0],
        S_final[..., 1],
        S_final[..., 2],
        S_final[..., 3],
        S_final[..., 4],
        S_final[..., 5],
    )
    vm2 = 0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3.0 * (
        s12**2 + s23**2 + s13**2
    )
    vm2 = jnp.clip(vm2, 0.0, jnp.inf)
    return jnp.mean(vm2)


def melt_loss_from_temperatures(
    temperatures, tctx, build_nodes, SMOOTH_ALPHA=20.0, margin=100.0
):
    """
    Smooth-max over time per node (same as in optimization), then deficit vs liquidus.
    Returns the UNWEIGHTED melt loss; multiply by MELT_W when plotting/summing.
    """
    # smooth max over time per node
    Tmax_soft = (
        jax.scipy.special.logsumexp(SMOOTH_ALPHA * temperatures, axis=0) / SMOOTH_ALPHA
    )
    deficit = jnp.maximum(tctx.liquidus + margin - Tmax_soft, 0.0) ** 2
    denom = jnp.maximum(jnp.sum(build_nodes), 1.0)
    return jnp.sum(deficit * build_nodes) / denom


def save_loss_components_plot(iteration, total_hist, stress_hist, meltw_hist, run_dir):
    import matplotlib.pyplot as plt, os

    iters = range(len(total_hist))
    plt.figure(figsize=(6, 4), dpi=140)
    plt.plot(iters, total_hist, label="Total loss", linewidth=2)
    plt.plot(iters, stress_hist, label="Stress component", linewidth=1.75)
    plt.plot(iters, meltw_hist, label=f"Melt component (weighted)", linewidth=1.75)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss breakdown")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(run_dir, f"loss_components_plot_{iteration:03d}.png")
    plt.savefig(out)
    plt.close()


# --------- main loss -----------
# weights you can tune
STRESS_W = 1.0  # weight on stress loss
MELT_W = 1  # weight on melt loss
SMOOTH_ALPHA = 30.0  # smooth-max sharpness; higher -> closer to max


def compute_losses(params):
    """Return (total_loss, stress_loss, weighted_melt_loss, control)."""
    control = expand_blocks_to_controls(params, power_on_steps, power_off_steps)

    T = simulate_temperature(control, tctx)  # (steps, n_n)
    S = simulate_mechanics(T, mctx)  # (T_m, n_e, n_q, 6)

    Sf = S[-1]
    s11, s22, s33, s12, s23, s13 = (
        Sf[..., 0],
        Sf[..., 1],
        Sf[..., 2],
        Sf[..., 3],
        Sf[..., 4],
        Sf[..., 5],
    )
    vm2 = 0.5 * (
        (s11 - s22) ** 2
        + (s22 - s33) ** 2
        + (s33 - s11) ** 2
        + 6.0 * (s12**2 + s23**2 + s13**2)
    )
    vm2 = jnp.clip(vm2, 0.0, jnp.inf)
    stress_loss = jnp.mean(vm2)

    # smooth-max over time per node
    Tmax_soft = logsumexp(SMOOTH_ALPHA * T, axis=0) / SMOOTH_ALPHA

    # melt deficit over build nodes
    margin = 100
    deficit = jnp.maximum(tctx.liquidus + margin - Tmax_soft, 0.0) ** 2
    denom = jnp.maximum(jnp.sum(build_nodes), 1.0)
    melt_loss = jnp.sum(deficit * build_nodes) / denom * MELT_W

    total_loss = stress_loss + melt_loss
    return total_loss, stress_loss, melt_loss, control


def main_function(params):
    total_loss, stress_loss, melt_w_loss, control = compute_losses(params)
    jax.debug.print(
        "loss: total={tot:.4e} | stress={sl:.4e} |  meltW={ml:.4e}",
        tot=total_loss,
        sl=stress_loss,
        ml=melt_w_loss,
    )
    return total_loss, control


# --- Put this near other plotting/IO helpers at top-level ---
def save_loss_components_plot(iteration, total_hist, stress_hist, meltw_hist, run_dir):
    import matplotlib.pyplot as plt

    iters = range(len(total_hist))
    plt.figure(figsize=(6, 4), dpi=140)
    plt.plot(iters, total_hist, label="Total loss", linewidth=2)
    plt.plot(iters, stress_hist, label="Stress component", linewidth=1.75)
    plt.plot(iters, meltw_hist, label=f"Melt component (x{MELT_W:g})", linewidth=1.75)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Loss breakdown")
    plt.legend()
    plt.tight_layout()
    out = os.path.join(run_dir, f"loss_components_plot_{iteration:03d}.png")
    plt.savefig(out)
    plt.close()


def grad_check(params, eps=1e-3, n_checks=5):
    """
    Compare JAX autodiff gradient of main_function to central finite differences.

    Args:
      params: 1D numpy array of initial control parameters.
      eps:    Finite-difference step size.
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
        p_plus = params.copy()
        p_plus[i] += eps
        p_minus = params.copy()
        p_minus[i] -= eps
        f_plus = float(loss_fn(p_plus))
        f_minus = float(loss_fn(p_minus))
        numeric_grad[i] = (f_plus - f_minus) / (2 * eps)

    table = []
    for i in range(n_checks):
        ag = autodiff_grad[i]
        ng = numeric_grad[i]
        err = abs(ag - ng)
        rel = err / (abs(ng) + 1e-8)
        table.append((i, ag, ng, err, rel))

    print(
        f"{'idx':>3} | {'autodiff':>12} | {'numeric':>12} | {'abs err':>10} | {'rel err':>10}"
    )
    print("-----+" + "-" * 14 + "+" + "-" * 14 + "+" + "-" * 12 + "+" + "-" * 12)
    for idx, ag, ng, err, rel in table:
        print(f"{idx:3d} | {ag:12.6e} | {ng:12.6e} | {err:10.2e} | {rel:10.2e}")

    return np.array(table)


def optimize_adam(
    params_init, num_iterations, run_dir, learning_rate=1e-3, build_animations=True
):
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

    make_animation_from_pattern(
        run_dir, "params_plot_*.png", out_stem="params_anim", fps=2
    )
    make_animation_from_pattern(
        run_dir, "control_plot_*.png", out_stem="control_anim", fps=2
    )
    make_animation_from_pattern(
        run_dir, "loss_history_plot_*.png", out_stem="loss_anim", fps=2
    )
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
    stress_history = []
    meltw_history = []

    # --- initial snapshot (iteration 0) ---
    p0_jax = jnp.asarray(params0, dtype=jnp.float32)
    L0, S0, M0, control0 = compute_losses(p0_jax)
    loss_history.append(float(L0))
    stress_history.append(float(S0))
    meltw_history.append(float(M0))
    control_history.append(np.array(control0))
    save_iter_artifacts(
        iteration=0,
        params_np=np.array(p0_jax),
        control_np=np.array(control0),
        loss_history=loss_history,
        run_dir=run_dir,
        power_on_steps=power_on_steps,
    )
    save_loss_components_plot(0, loss_history, stress_history, meltw_history, run_dir)

    eval_cache = {"last_loss": float(L0), "last_control": np.array(control0), "iter": 1}

    def loss_only(p_jax):
        # keep scalar for optimizer
        return main_function(p_jax)[0]

    loss_and_grad = jax.jit(jax.value_and_grad(loss_only))

    def fun_and_grad(x_np):
        x_jax = jnp.asarray(x_np, dtype=jnp.float32)
        val, grad = loss_and_grad(x_jax)
        control = expand_blocks_to_controls(x_jax, power_on_steps, power_off_steps)
        eval_cache["last_loss"] = float(val)
        eval_cache["last_control"] = np.asarray(control)
        return float(val), np.asarray(grad, dtype=np.float64)

    def cb(xk):
        it = eval_cache["iter"]
        # Log scalar histories
        loss_history.append(eval_cache["last_loss"])
        control_history.append(eval_cache["last_control"])

        # Compute components for this iterate xk
        Lk, Sk, Mk, _ = compute_losses(jnp.asarray(xk, dtype=jnp.float32))
        stress_history.append(float(Sk))
        meltw_history.append(float(Mk))

        save_iter_artifacts(
            iteration=it,
            params_np=np.array(xk, dtype=np.float32),
            control_np=eval_cache["last_control"],
            loss_history=loss_history,
            run_dir=run_dir,
            power_on_steps=power_on_steps,
        )
        save_loss_components_plot(
            it, loss_history, stress_history, meltw_history, run_dir
        )
        print(f"[LBFGS] iter {it:03d}  loss={loss_history[-1]:.6e}")
        eval_cache["iter"] = it + 1

    bounds = [(CTRL_MIN, CTRL_MAX)] * int(N_BLOCKS)
    res = spo.minimize(
        fun_and_grad,
        x0=params0,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options=dict(
            maxiter=int(num_iterations), gtol=1e-6, ftol=1e-10, maxcor=10, maxls=60
        ),
        callback=cb,
    )

    print(f"[LBFGS] status={res.status}  message={res.message}")
    trained_params = res.x.astype(np.float32)

    # (optional) save consolidated "latest" files
    np.save(os.path.join(run_dir, "params_bfgs_latest.npy"), trained_params)
    np.save(os.path.join(run_dir, "loss_bfgs_latest.npy"), np.array(loss_history))

    # After: trained_params, loss_history, control_history = optimize_bfgs(...)
    # Build per-plot animations
    make_animation_from_pattern(
        run_dir, "params_plot_*.png", out_stem="params_anim", fps=2
    )
    make_animation_from_pattern(
        run_dir, "control_plot_*.png", out_stem="control_anim", fps=2
    )
    make_animation_from_pattern(
        run_dir, "loss_history_plot_*.png", out_stem="loss_anim", fps=2
    )
    make_animation_from_pattern(
        run_dir, "loss_components_plot_*.png", out_stem="loss_components_anim", fps=2
    )

    # Optional: single side-by-side dashboard per iteration (params | control | loss)
    make_iteration_dashboard(run_dir, out_stem="dashboard_anim", fps=2)

    return (
        trained_params,
        np.array(loss_history),
        np.array(control_history, dtype=object),
    )


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
            num_iterations=num_iters,  # <- use CLI
            run_dir=run_dir,
            learning_rate=learning_rate,
        )

    elif mode == "bfgs":
        print("Running bfgs optimization...")
        trained_params, loss_history, control_history = optimize_bfgs(
            params_init=params,
            num_iterations=num_iters,  # <- use CLI
            run_dir=run_dir,
        )

        # --- After BFGS finishes: write metrics JSON similar to baseline ---
        # Get final params from the optimizer result (adjust name if yours differs)
        final_params = np.asarray(
            trained_params, dtype=np.float32
        )  # <-- 'result' is your scipy minimize output

        # Rebuild the control from block params
        final_control = expand_blocks_to_controls(
            jnp.asarray(final_params), power_on_steps, power_off_steps
        )

        # Forward sims to mirror baseline metric calculation
        temperatures_f = simulate_temperature(final_control, tctx)
        S_seq_f, U_seq_f = simulate_mechanics_forward(
            temperatures_f, mctx
        )  # returns (S_seq, U_seq) :contentReference[oaicite:2]{index=2}
        S_last_f = S_seq_f[-1]  # final snapshot from forward mechanics

        # Loss components
        stress_loss = float(stress_loss_from_S_final(S_last_f))  # mean VM^2
        melt_loss = float(
            melt_loss_from_temperatures(
                temperatures_f,
                tctx,
                build_nodes,
                SMOOTH_ALPHA=SMOOTH_ALPHA,
                margin=100.0,
            )
        )
        total_loss = stress_loss + MELT_W * melt_loss

        # Von Mises metrics (final snapshot)
        # (If you already have von_mises_from_S, you can use that directly; else reuse the math above)
        s11, s22, s33, s12, s23, s13 = (
            S_last_f[..., 0],
            S_last_f[..., 1],
            S_last_f[..., 2],
            S_last_f[..., 3],
            S_last_f[..., 4],
            S_last_f[..., 5],
        )
        vm2 = 0.5 * ((s11 - s22) ** 2 + (s22 - s33) ** 2 + (s33 - s11) ** 2) + 3.0 * (
            s12**2 + s23**2 + s13**2
        )
        vm = jnp.sqrt(jnp.maximum(vm2, 0.0))
        avg_vm = float(jnp.mean(vm))
        p95_vm = float(jnp.percentile(vm, 95.0))
        max_vm = float(jnp.max(vm))

        # Save JSON right next to other BFGS artifacts
        metrics_path = os.path.join(run_dir, "bfgs_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(
                {
                    "loss_total": total_loss,
                    "loss_stress": stress_loss,
                    "loss_melt_weighted": MELT_W * melt_loss,
                    "loss_melt_unweighted": melt_loss,
                    "avg_vm_final": avg_vm,
                    "p95_vm_final": p95_vm,
                    "max_vm_final": max_vm,
                },
                f,
                indent=2,
            )
        print(f"[bfgs] Wrote metrics -> {metrics_path}")

    elif mode == "forward":
        print("Running forward simulation...")
        baseline_dir = os.path.join(work_dir, "forward")
        os.makedirs(baseline_dir, exist_ok=True)

        found_any = False
        for src in ["adam", "bfgs"]:
            try:
                ctrl_path, src_run_dir = latest_control_under(str(work_dir), src)
            except FileNotFoundError as e:
                print(f"[forward] Skip {src}: {e}")
                continue

            found_any = True
            print(f"[forward] Using control from {src}: {ctrl_path}")

            control = jnp.array(np.load(ctrl_path), dtype=jnp.float32)
            if control.shape[0] < steps:
                control = jnp.concatenate(
                    [control, jnp.zeros((steps - control.shape[0],), control.dtype)],
                    axis=0,
                )
            elif control.shape[0] > steps:
                control = control[:steps]

            # simulate
            temperatures = simulate_temperature(control, tctx)
            S_seq, U_seq = simulate_mechanics_forward(temperatures, mctx)

            save_vtk(
                temperatures,
                S_seq,
                U_seq,
                elements,
                Bip_ele,
                nodes,
                element_birth,
                node_birth,
                dt,
                run_dir=baseline_dir,
                keyword="forward",
            )

            # 3b) Compute loss components (single 'iteration' baseline snapshot)
            S_last = S_seq[-1]
            stress_loss = float(stress_loss_from_S_final(S_last))  # mean VM^2
            melt_loss = float(
                melt_loss_from_temperatures(
                    temperatures,
                    tctx,
                    build_nodes,
                    SMOOTH_ALPHA=SMOOTH_ALPHA,
                    margin=100.0,
                )
            )
            total_loss = stress_loss + MELT_W * melt_loss

            print(
                f"[baseline_avg] loss: total={total_loss:.4e} | stress={stress_loss:.4e} | meltW={(MELT_W*melt_loss):.4e}"
            )

            # Histories (single point to match plotting API)
            loss_history = [total_loss]
            stress_history = [stress_loss]
            meltw_history = [MELT_W * melt_loss]

            # 3c) (Keep your VM metrics if you like them too)
            vm = jnp.sqrt(
                jnp.maximum(0.0, 2.0 * stress_loss)
            )  # not used; you already compute below more directly
            vm_final = von_mises_from_S(S_last)
            avg_vm = float(jnp.mean(vm_final))
            max_vm = float(jnp.max(vm_final))
            p95_vm = float(jnp.percentile(vm_final, 95.0))
            print(
                f"[baseline_avg] Von Mises (final step): mean={avg_vm:.4e}, p95={p95_vm:.4e}, max={max_vm:.4e}"
            )

            # Save metrics for easy comparison later
            with open(os.path.join(baseline_dir, "baseline_metrics.json"), "w") as f:
                json.dump(
                    {
                        "loss_total": total_loss,
                        "loss_stress": stress_loss,
                        "loss_melt_weighted": MELT_W * melt_loss,
                        "loss_melt_unweighted": melt_loss,
                        "avg_vm_final": avg_vm,
                        "p95_vm_final": p95_vm,
                        "max_vm_final": max_vm,
                    },
                    f,
                    indent=2,
                )

            if not found_any:
                print("[forward] No controls found under run_dir/adam or run_dir/bfgs.")

    elif mode == "baseline":
        print("Running baseline simulation...")
        baseline_dir = os.path.join(work_dir, "baseline")
        os.makedirs(baseline_dir, exist_ok=True)

        control = jnp.ones((power_on_steps,), dtype=jnp.float32)
        control = jnp.concatenate(
            [control, jnp.zeros((steps - control.shape[0],), control.dtype)], axis=0
        )
        temperatures = simulate_temperature(control, tctx)
        S_seq, U_seq = simulate_mechanics_forward(temperatures, mctx)
        save_vtk(
            temperatures,
            S_seq,
            U_seq,
            elements,
            Bip_ele,
            nodes,
            element_birth,
            node_birth,
            dt,
            run_dir=baseline_dir,
            keyword="baseline",
        )

        # 3b) Compute loss components (single 'iteration' baseline snapshot)
        S_last = S_seq[-1]
        stress_loss = float(stress_loss_from_S_final(S_last))  # mean VM^2
        melt_loss = float(
            melt_loss_from_temperatures(
                temperatures, tctx, build_nodes, SMOOTH_ALPHA=SMOOTH_ALPHA, margin=100.0
            )
        )
        total_loss = stress_loss + MELT_W * melt_loss

        print(
            f"[baseline_avg] loss: total={total_loss:.4e} | stress={stress_loss:.4e} | meltW={(MELT_W*melt_loss):.4e}"
        )

        # Histories (single point to match plotting API)
        loss_history = [total_loss]
        stress_history = [stress_loss]
        meltw_history = [MELT_W * melt_loss]

        # 3c) (Keep your VM metrics if you like them too)
        vm = jnp.sqrt(
            jnp.maximum(0.0, 2.0 * stress_loss)
        )  # not used; you already compute below more directly
        vm_final = von_mises_from_S(S_last)
        avg_vm = float(jnp.mean(vm_final))
        max_vm = float(jnp.max(vm_final))
        p95_vm = float(jnp.percentile(vm_final, 95.0))
        print(
            f"[baseline_avg] Von Mises (final step): mean={avg_vm:.4e}, p95={p95_vm:.4e}, max={max_vm:.4e}"
        )

        # Save metrics for easy comparison later
        with open(os.path.join(baseline_dir, "baseline_metrics.json"), "w") as f:
            json.dump(
                {
                    "loss_total": total_loss,
                    "loss_stress": stress_loss,
                    "loss_melt_weighted": MELT_W * melt_loss,
                    "loss_melt_unweighted": melt_loss,
                    "avg_vm_final": avg_vm,
                    "p95_vm_final": p95_vm,
                    "max_vm_final": max_vm,
                },
                f,
                indent=2,
            )

    elif mode == "baseline_avg":
        print("Running baseline forward simulation from mean of final BFGS control...")
        baseline_dir = os.path.join(work_dir, "baseline_avg")
        os.makedirs(baseline_dir, exist_ok=True)

        # 1) Load latest BFGS control path
        try:
            ctrl_path, src_run_dir = latest_control_under(str(work_dir), "bfgs")
        except FileNotFoundError as e:
            raise SystemExit(f"[baseline_avg] No BFGS control found: {e}")

        print(f"[baseline_avg] Using final BFGS control: {ctrl_path}")
        control_opt = jnp.array(np.load(ctrl_path), dtype=jnp.float32)

        # Pad/trim to match current steps (mirrors your forward branch)
        if control_opt.shape[0] < steps:
            control_opt = jnp.concatenate(
                [
                    control_opt,
                    jnp.zeros((steps - control_opt.shape[0],), control_opt.dtype),
                ],
                axis=0,
            )
        elif control_opt.shape[0] > steps:
            control_opt = control_opt[:steps]

        # 2) Baseline = constant mean during ON, zero during OFF
        control_base, const_val = constant_on_from_control(
            control_opt, power_on_steps, power_off_steps
        )
        print(
            f"[baseline_avg] Constant power value (ON window mean) = {float(const_val):.6f}"
        )

        # 3) Forward sims
        temperatures = simulate_temperature(control_base, tctx)
        S_seq, U_seq = simulate_mechanics_forward(temperatures, mctx)

        save_vtk(
            temperatures,
            S_seq,
            U_seq,
            elements,
            Bip_ele,
            nodes,
            element_birth,
            node_birth,
            dt,
            run_dir=baseline_dir,
            keyword="baseline_avg",
        )

        # 3b) Compute loss components (single 'iteration' baseline snapshot)
        S_last = S_seq[-1]
        stress_loss = float(stress_loss_from_S_final(S_last))  # mean VM^2
        melt_loss = float(
            melt_loss_from_temperatures(
                temperatures, tctx, build_nodes, SMOOTH_ALPHA=SMOOTH_ALPHA, margin=100.0
            )
        )
        total_loss = stress_loss + MELT_W * melt_loss

        print(
            f"[baseline_avg] loss: total={total_loss:.4e} | stress={stress_loss:.4e} | meltW={(MELT_W*melt_loss):.4e}"
        )

        # Histories (single point to match plotting API)
        loss_history = [total_loss]
        stress_history = [stress_loss]
        meltw_history = [MELT_W * melt_loss]

        # 3c) (Keep your VM metrics if you like them too)
        vm = jnp.sqrt(
            jnp.maximum(0.0, 2.0 * stress_loss)
        )  # not used; you already compute below more directly
        vm_final = von_mises_from_S(S_last)
        avg_vm = float(jnp.mean(vm_final))
        max_vm = float(jnp.max(vm_final))
        p95_vm = float(jnp.percentile(vm_final, 95.0))
        print(
            f"[baseline_avg] Von Mises (final step): mean={avg_vm:.4e}, p95={p95_vm:.4e}, max={max_vm:.4e}"
        )

        # Save metrics for easy comparison later
        with open(os.path.join(baseline_dir, "baseline_metrics.json"), "w") as f:
            json.dump(
                {
                    "loss_total": total_loss,
                    "loss_stress": stress_loss,
                    "loss_melt_weighted": MELT_W * melt_loss,
                    "loss_melt_unweighted": melt_loss,
                    "avg_vm_final": avg_vm,
                    "p95_vm_final": p95_vm,
                    "max_vm_final": max_vm,
                },
                f,
                indent=2,
            )

    t_end = time.time()
    print(f"Total Time: {t_end - t_start:.2f} seconds")
