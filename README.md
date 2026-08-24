# Differentiable Thermomechanical AM

JAX-based, end-to-end differentiable thermomechanical simulation of laser powder directed energy deposition (DED), used to inversely optimize a time-series laser power schedule that minimizes post-build residual stress while maintaining adequate melting. 

The pipeline chains a moving-laser-source thermal FEA solver to an elastoplastic mechanical FEA solver, both written in JAX, and optimizes the laser power schedule (Adam or L-BFGS-B) against a two-component stress + melt objective via automatic differentiation (custom adjoint through the implicit mechanics Newton solve).

## Repository Layout
- `Optimizer_main.py` — primary entrypoint for running optimization/forward/baseline jobs.
- `includes/` — thermal/mechanics JAX solvers (`thermal.py`, `mech.py`), data loading (`data_loader.py`), visualization/IO helpers (`utils.py`), and a separate, GPU-only legacy mesh/thermal preprocessing tool (`preprocessor.py`, see below).
- `materials/` — temperature-dependent material property tables (Ti-6Al-4V, IN718, SS316L); only the `TI64_*_Debroy.txt` files are currently read by `Optimizer_main.py`.
- `preprocessed/` — mesh/toolpath source files (`<base>.inp`, `<base>.k`, `<base>.crs`) and the arrays derived from them (`<base>_preprocessed/`), one set per case.
- `results/` — run outputs (ignored by git), organized by `<base>_*` folders and optimizer mode.
- `experimental/` — currently just `preprocess.py`, the CLI wrapper that regenerates `preprocessed/<base>_preprocessed/` from raw mesh/toolpath files via `includes/preprocessor.py`. Everything else that used to live here (tuning-history forks of `Optimizer_main.py`, a stale float64 testbed, a duplicate melt-diagnostic script) was cut during a 2026 refresh as dead or fully superseded.

## Setup

Two conda environments are provided:

- **`environment.yml`** — Linux + NVIDIA GPU (CUDA 12.x driver). Covers the full pipeline, including the GPU-only preprocessing step (`cupy`, `numba`).
- **`environment-cpu.yml`** — CPU-only. Enough to run `Optimizer_main.py` against the four datasets already checked into `preprocessed/*_preprocessed/` (slower, but no GPU required). Cannot run the preprocessing step.

```bash
conda env create -f environment.yml       # or environment-cpu.yml
conda activate diffmech-am                # or diffmech-am-cpu
```

Python 3.12 is a hard requirement (not just a recommendation) — `includes/utils.py` type hints and the pinned `jax`/`jaxlib` release both need it. Optional: `ffmpeg` on your `PATH` enables MP4 exports from animations.

> `environment.yml` was authored without access to GPU hardware and has not been dependency-solved or run — validate it on a real Linux + CUDA 12 machine before relying on it. `environment-cpu.yml` has been created and smoke-tested end to end (see "Running the main script" below).

## Running the main script
1) Pick a dataset name (`base_name`) that matches a folder in `preprocessed/` (e.g., `1_stsl_preprocessed`) and its toolpath file (`preprocessed/1_stsl.crs`). Update the `base_name` variable near the top of `Optimizer_main.py`. Ready-to-run datasets: `1_stsl`, `2_stml`, `3_mtsl`, `4_mtml` (each has a complete `_preprocessed/` folder). `1x5` and `4_mtml_half` are leftover partial mesh files with no matching `.crs`/`_preprocessed/` data and cannot be run as-is.
2) Launch an optimization run (writes under `results/<base>_..._<n_blocks>params_gradcheck/<mode>/`):
   ```bash
   CUDA_VISIBLE_DEVICES=0 python Optimizer_main.py bfgs --iters 60 --n-blocks 10 --gpu 0
   ```
   Modes: `adam`, `bfgs`, `gradcheck`, `forward`, `baseline`, `baseline_avg`.
   Flags: `--iters` (optimizer iterations, default 10), `--n-blocks` (number of piecewise-linear power control knots, default 10), `--gpu` (CUDA device id), `--tag` (optional run-folder label), `--ts` (add a timestamp subfolder).
3) To reproduce the case studies:

   | Case | `base_name` | `--n-blocks` | `--iters` |
   |---|---|---|---|
   | Single-track, single-layer | `1_stsl` | 10 | 60 |
   | Single-track, multi-layer (3 layers) | `2_stml` | 30 | 60* |
   | Multi-track, single-layer (L-shape) | `3_mtsl` | 10 | 81 |
   | Multi-track, multi-layer (L-block) | `4_mtml` | 30 | 100 |

   `--n-blocks` follows a ~10-knots-per-layer convention. \*`2_stml`'s `--iters` is a starting-point suggestion, not a validated/tuned value like the other three rows — check the loss curve (`bfgs_metrics.json` / the loss animation) and increase if it hasn't plateaued.

4) `forward`/`baseline`/`baseline_avg` modes reuse the latest controls produced by `adam`/`bfgs` and dump VTK/PNG artifacts to the same run folder — **pass the same `--n-blocks` (and `base_name`) used for that run**, since the run folder's name encodes the param count and won't otherwise resolve to the right directory.

## Preprocessing (GPU-only, optional)
`experimental/preprocess.py` regenerates a `preprocessed/<base>_preprocessed/` folder from raw `.inp`/`.crs` mesh/toolpath files, via `includes/preprocessor.py` (`cupy`/`numba`-based, requires `environment.yml`/CUDA). `base_name` is hardcoded near the top of the script and must be edited to point at a different case. Not needed if you're only running the four datasets already provided.

## Data expectations
- Mesh/toolpath inputs live in `preprocessed/` (`<base>.inp`, `<base>.k`, `<base>.crs`).
- Preprocessed arrays (nodes/elements/birth/surface flux) live in `preprocessed/<base>_preprocessed/`.
- Material property tables are read from `materials/*.txt`.

## Known limitations
- The mechanics solve is downsampled to every 10th thermal step (`stride=10` in `includes/mech.py`), and `includes/utils.py:save_vtk` independently hardcodes a matching `dt=0.1`/`×10` step-index calculation for VTK export timing. These are currently self-consistent (thermal `dt=0.01` × stride 10 = 0.1) but not derived from each other — changing one without the other will silently desync VTK export timestamps from the real elapsed time.
- Memory grows as `O(N_timesteps × N_mesh)` since the differentiable simulation unrolls gradients through time without checkpointing — this bounds how large a case can be optimized on a single GPU
- `includes/preprocessor.py`'s `write_parameters` embeds a legacy IN718 material template (`*MAT_THERMAL_ISOTROPIC_TD`, `*GAUSS_LASER 400 1.12 0.4`) left over from an earlier case study; the JAX pipeline reads Ti-6Al-4V properties directly from `materials/TI64_*_Debroy.txt` and ignores this template, but it's a placeholder to update if the `.k`-file material block is ever wired back in.

## Housekeeping
- `results/` is ignored by git; run artifacts stay contained there.
- If you run from outside the repo root, path resolution is handled via `includes.utils.get_project_root()` so long as the `preprocessed/` marker directory is present.
