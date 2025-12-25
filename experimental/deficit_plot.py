import argparse
import os
from pathlib import Path

# --- CLI: parse before importing jax so we can set GPU ---
parser = argparse.ArgumentParser(
    description="Forward thermal runs + melt deficit diagnostics"
)
parser.add_argument(
    "--gpu", type=str, default="0", help="CUDA device id (string, e.g. '0')"
)
parser.add_argument(
    "--base-name",
    type=str,
    default="1_stsl",
    help="Base name used for preprocessed data and toolpath (e.g. '1_stsl')",
)
parser.add_argument(
    "--work-dir",
    type=str,
    default=None,
    help="Root work directory. If None, defaults to './{base}_bfgsbound_10params_gradcheck'",
)
parser.add_argument(
    "--src",
    type=str,
    default="bfgs",
    choices=["bfgs", "adam"],
    help="Which optimization run to load the optimized control from",
)
parser.add_argument(
    "--margin",
    type=float,
    default=100.0,
    help="Temperature margin added to liquidus in melt loss (K)",
)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu  # must be set before importing jax

import numpy as np
import jax
import jax.numpy as jnp
from includes.data_loader import load_data
from includes.thermal import ThermContext, simulate_temperature
from includes.utils import latest_control_under, get_project_root
import pyvista as pv

jax.config.update("jax_enable_x64", True)

PROJECT_ROOT = get_project_root()
DATA_ROOT = PROJECT_ROOT / "preprocessed"
RESULTS_ROOT = PROJECT_ROOT / "results"

# --- Setup paths ---
base_name = args.base_name
if args.work_dir is None:
    work_dir = RESULTS_ROOT / f"{base_name}_bfgsbound_20params_fix"
else:
    work_dir = Path(args.work_dir)
    if not work_dir.is_absolute():
        work_dir = PROJECT_ROOT / work_dir
out_root = work_dir / "forward_melt_diag"
os.makedirs(out_root, exist_ok=True)

print(f"[io] work_dir = {work_dir}")
print(f"[io] out_root = {out_root}")
print(f"[cfg] base_name = {base_name}, src = {args.src}, margin = {args.margin}")

# --- Load mesh + toolpath data (same as MWE_bound.py) ---
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

# Time grid and ON/OFF windows
power_on_time = float(toolpath_name.read_text().strip().splitlines()[-2].split()[0])
print(f"[time] endTime from toolpath file   : {endTime}")
print(f"[time] power_on_time from toolpath  : {power_on_time}")

steps = int(endTime / dt) + 1
power_on_steps = int(power_on_time / dt)
power_off_steps = steps - power_on_steps
print(f"[time] Total steps = {steps}, ON = {power_on_steps}, OFF = {power_off_steps}")

n_n = len(nodes)
n_e = len(elements)
n_q = 8  # quadrature points per element

# --- Material & heat transfer properties (Ti64, Debroy data) ---
ambient = 300.0
density = 0.0044
cp_val = 0.714  # in your consistent units
cond_val = 0.01780
Qin = 500.0 * 0.4  # absorbed laser power at control=1.0
base_power = Qin
r_beam = 1.12
h_conv = 5e-5
emissivity = 0.2
solidus = 1878.0
liquidus = 1928.0
latent = 286.0 / (liquidus - solidus)
conds = jnp.ones((n_e, 8)) * cond_val
stefan_boltz = 5.670374419e-8  # W*m^-2*K^-4 (Stefan-Boltzmann)

# Build region mask (same as MWE_bound.py)
build_nodes = (nodes[:, 2] > 0.1).astype(jnp.float32)
N_nodes = jnp.maximum(jnp.sum(build_nodes), 1.0)

print(f"[geo] n_n = {n_n}, n_e = {n_e}, build_nodes = {int(N_nodes)}")

# Laser activation
laser_loc = jnp.array(toolpath)
laser_on = jnp.array(state)

# --- Thermo material models (Ti64, temp-dependent) ---
poisson = 0.3  # only used in mechanics, kept for completeness
a1 = 10000.0  #   "
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

# --- Thermal context (no mechanics needed here) ---
tctx = ThermContext(
    # mesh
    n_n=int(n_n),
    n_e=int(n_e),
    n_q=int(n_q),
    elements=elements,
    nodes=nodes,
    Nip_ele=Nip_ele,
    Bip_ele=Bip_ele,
    # surfaces
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
    BOT_NODES=(nodes[:, 2] < -0.9),  # only used for mech, harmless here
)

# --- Helpers ---------------------------------------------------------------


def expand_blocks_to_controls(block_controls, power_on_steps, power_off_steps):
    """
    Expand N_BLOCKS piecewise-constant controls to per-timestep control over the ON window,
    then append zeros over the OFF window.
    """
    block_controls = jnp.asarray(block_controls, dtype=jnp.float32)
    n_blocks = block_controls.shape[0]
    stride = power_on_steps // n_blocks
    knots = jnp.arange(0, power_on_steps, stride)
    t = jnp.arange(power_on_steps)
    control_on_interp = jnp.interp(t, knots, block_controls)
    control_full = jnp.concatenate(
        [control_on_interp, jnp.zeros((power_off_steps,), dtype=jnp.float32)],
        axis=0,
    )
    return control_full


def constant_on_control(value, power_on_steps, power_off_steps):
    """Return control(t) = value during laser ON, 0 after."""
    control_on = jnp.full((power_on_steps,), value, dtype=jnp.float32)
    control_off = jnp.zeros((power_off_steps,), dtype=jnp.float32)
    return jnp.concatenate([control_on, control_off], axis=0)


def melt_deficit_hardmax(temperatures, tctx, build_nodes, margin):
    """
    Hard-max over time per node, then squared deficit vs (liquidus + margin) on build_nodes.
    Returns Tmax (per node) and deficit (per node).
    """
    T_np = np.asarray(temperatures)  # (steps, n_n)
    Tmax = T_np.max(axis=0)
    threshold = float(tctx.liquidus + margin)
    deficit = np.maximum(threshold - Tmax, 0.0)
    build_np = np.asarray(build_nodes, dtype=np.float32)
    deficit *= build_np
    return Tmax, deficit


def save_melt_deficit_vtu(nodes, elements, Tmax, deficit, out_path):
    """
    Write a simple VTU with nodal fields:
      - Tmax           : hard-max temperature per node
      - melt_deficit   : squared deficit per node (0 outside build)
    """
    nodes_np = np.asarray(nodes, dtype=float)
    elements_np = np.asarray(elements, dtype=np.int64)
    n_e, n_p = elements_np.shape
    if n_p != 8:
        raise ValueError(f"Expected 8-node bricks, got {n_p} nodes/element")

    # VTK 'cells' array: [8, n0, ..., n7, 8, n0, ..., n7, ...]
    cells = np.hstack([np.full((n_e, 1), 8, dtype=np.int64), elements_np]).ravel()

    # HEXAHEDRON cell type = 12
    cell_types = np.full(n_e, 12, dtype=np.uint8)

    grid = pv.UnstructuredGrid(cells, cell_types, nodes_np)
    grid.point_data["Tmax"] = np.asarray(Tmax, dtype=np.float32)
    grid.point_data["melt_deficit"] = np.asarray(deficit, dtype=np.float32)
    grid.save(out_path)
    print(f"[vtk] Wrote {out_path}")


def run_case(name, control):
    """Simulate temperature for a given control and dump melt deficit diagnostics."""
    print(f"\n=== Case: {name} ===")
    control = jnp.asarray(control, dtype=jnp.float32)
    assert control.shape[0] == steps

    temperatures = simulate_temperature(control, tctx)  # (steps, n_n)

    # melt diagnostics
    Tmax, deficit = melt_deficit_hardmax(
        temperatures, tctx, build_nodes, margin=args.margin
    )
    n_def = int((deficit > 0.0).sum())
    n_build = int(np.asarray(build_nodes).sum())
    print(f"[{name}] Nodes in build region        : {n_build}")
    print(f"[{name}] Nodes with nonzero deficit   : {n_def}")
    if n_build > 0:
        print(f"[{name}] Fraction with deficit        : {n_def / n_build:.3f}")

    # save NPYs and VTU
    case_dir = out_root / name
    os.makedirs(case_dir, exist_ok=True)
    np.save(case_dir / f"Tmax_{name}.npy", Tmax)
    np.save(case_dir / f"melt_deficit_{name}.npy", deficit)
    save_melt_deficit_vtu(
        nodes, elements, Tmax, deficit, case_dir / f"melt_deficit_{name}.vtu"
    )


# --- Build controls for the three cases -----------------------------------

# 1) Constant power = 1 during ON, 0 after
control_const1 = constant_on_control(1.0, power_on_steps, power_off_steps)

# 2) Optimized control: load latest control_*.npy from chosen src (bfgs/adam)
print("\n[load] Searching for optimized control...")
ctrl_path, src_run_dir = latest_control_under(str(work_dir), args.src)
print(f"[load] Using control from {args.src}: {ctrl_path}")

control_opt = np.load(ctrl_path).astype(np.float32)
if control_opt.shape[0] < steps:
    pad = np.zeros((steps - control_opt.shape[0],), dtype=np.float32)
    control_opt = np.concatenate([control_opt, pad], axis=0)
elif control_opt.shape[0] > steps:
    control_opt = control_opt[:steps]
control_opt = jnp.asarray(control_opt, dtype=jnp.float32)

# 3) Constant power = mean of optimized power over ON window
on_window = np.asarray(control_opt[:power_on_steps])
avg_power = float(on_window.mean()) if on_window.size > 0 else 0.0
print(f"[ctrl] Mean optimized power on ON window = {avg_power:.6f}")
control_const_avg = constant_on_control(avg_power, power_on_steps, power_off_steps)

# --- Run all three cases ---------------------------------------------------
run_case("const1", control_const1)
run_case("optimized", control_opt)
run_case("const_avg", control_const_avg)

print(
    "\nDone. Open the '*.vtu' files in ParaView and color by 'melt_deficit' to see where melt loss comes from."
)
