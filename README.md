# Differentiable Thermomechanical AM

JAX-based thermo-mechanical simulation and power-schedule optimization for metal additive manufacturing. The code supports differentiable temperature/mechanics solves and optimizers (Adam, L-BFGS-B) to tune per-layer laser power.

## Repository Layout
- `Optimizer_main.py` - primary entrypoint for running optimization/forward/baseline jobs.
- `includes/` - thermal/mechanics solvers, data loading, and utilities.
- `materials/` - temperature-dependent material property tables.
- `preprocessed/` - meshes, birth schedules, and toolpaths (`<base>.crs`) organized per case (`<base>_preprocessed/`).
- `results/` - run outputs (ignored by git) organized by `<base>_*` folders and optimizer mode.
- `experimental/` - archived experiments (bounded variants, deficit diagnostics, preprocessing helpers, x64 testbed).

## Setup
1) Python 3.10+ is recommended. A GPU-enabled JAX build is strongly advised for performance (`jax`/`jaxlib` install should match your CUDA version).
2) Install dependencies (adjust if you already have GPU-specific `jaxlib`):
   ```bash
   pip install jax jaxlib optax numpy scipy matplotlib imageio pillow pandas pyvista vtk imageio-ffmpeg
   ```
   Optional: `ffmpeg` on your PATH enables MP4 exports from animations.

## Running the main script
1) Pick a dataset name (`base_name`) that matches a folder in `preprocessed/` (e.g., `1_stsl_preprocessed`) and its toolpath file (`preprocessed/1_stsl.crs`). Update the `base_name` variable near the top of `Optimizer_main.py` if needed.
2) Launch an optimization run (writes under `results/<base>_.../<mode>/`):
   ```bash
   CUDA_VISIBLE_DEVICES=0 python Optimizer_main.py bfgs --iters 10 --gpu 0
   ```
   Modes: `adam`, `bfgs`, `gradcheck`, `forward`, `baseline`, `baseline_avg`.
3) Forward/baseline modes reuse the latest controls produced by `adam`/`bfgs` and dump VTK/PNG artifacts to the same run folder.

## Experimental utilities
- `experimental/deficit_plot.py` - run forward thermal cases and export melt-deficit VTUs.
- `experimental/MWE*.py` - legacy/variant optimizers with different bounds/weights.
- `experimental/preprocess.py` - helper to build the `preprocessed/<base>_preprocessed` arrays from mesh/toolpath files.
- `experimental/x64_test.py` - float64/numerics stress test.

## Data expectations
- Mesh/toolpath inputs live in `preprocessed/` (`<base>.inp`, `<base>.k`, `<base>.crs`).
- Preprocessed arrays (nodes/elements/birth/surface flux) live in `preprocessed/<base>_preprocessed/`.
- Material property tables are read from `materials/*.txt`.

## Housekeeping
- `results/` is ignored by git; run artifacts stay contained there.
- If you run from outside the repo root, path resolution is handled via `includes.utils.get_project_root()` so long as the `preprocessed/` marker directory is present.
