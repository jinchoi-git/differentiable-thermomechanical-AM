# Differentiable Thermomechanical AM

JAX-based, end-to-end differentiable thermomechanical simulation of laser powder directed energy deposition (DED). Chains a moving-laser-source thermal FEA solver to an elastoplastic mechanical FEA solver (both in JAX) and optimizes a time-series laser power schedule (Adam or L-BFGS-B) against a stress + melt objective via automatic differentiation, to minimize post-build residual stress while maintaining adequate melting.

## Layout
- `Optimizer_main.py` — entrypoint for optimization/forward/baseline runs.
- `includes/` — JAX thermal/mechanics solvers (`thermal.py`, `mech.py`), data loading (`data_loader.py`), viz/IO helpers (`utils.py`), and a GPU-only legacy mesh/thermal preprocessor (`preprocessor.py`).
- `materials/` — material property tables; only `TI64_*_Debroy.txt` is currently read.
- `preprocessed/` — mesh/toolpath sources (`<base>.inp/.k/.crs`) and derived arrays (`<base>_preprocessed/`).
- `results/` — run outputs (gitignored).
- `experimental/` — `preprocess.py`, the only tool that regenerates `preprocessed/<base>_preprocessed/` from raw mesh/toolpath files.

## Setup

```bash
conda env create -f environment.yml       # Linux + NVIDIA GPU (CUDA 12.x)
# or
conda env create -f environment-cpu.yml   # CPU-only, can't run preprocessing
conda activate diffmech-am                # or diffmech-am-cpu
```

Requires Python 3.12. Optional: `ffmpeg` on `PATH` for MP4 animation export.

> `environment.yml` hasn't been solved/tested on real GPU hardware — validate before relying on it. `environment-cpu.yml` has been smoke-tested end to end.

## Running

1. Set `base_name` near the top of `Optimizer_main.py` to a dataset in `preprocessed/`: `1_stsl`, `2_stml`, `3_mtsl`, or `4_mtml` (`1x5` and `4_mtml_half` are incomplete leftovers, not runnable).
2. Run:
   ```bash
   CUDA_VISIBLE_DEVICES=0 python Optimizer_main.py bfgs --iters 60 --n-blocks 10 --gpu 0
   ```
   Modes: `adam`, `bfgs`, `gradcheck`, `forward`, `baseline`, `baseline_avg`.
   Flags: `--iters` (default 10), `--n-blocks` — piecewise-linear power control knots, default 10 (`forward`/`baseline`/`baseline_avg` must reuse the same `--n-blocks`/`base_name` as the run they read from, since it's baked into the run-folder name), `--gpu`, `--tag`, `--ts` (timestamp subfolder).
3. Known-good configs per dataset:

   | `base_name` | Description | `--n-blocks` | `--iters` |
   |---|---|---|---|
   | `1_stsl` | single-track, single-layer | 10 | 60 |
   | `2_stml` | single-track, 3 layers | 30 | 60* |
   | `3_mtsl` | multi-track, single-layer (L-shape) | 10 | 81 |
   | `4_mtml` | multi-track, 3 layers (L-block) | 30 | 100 |

   `--n-blocks` follows ~10 knots/layer. \*`2_stml`'s iters is an unvalidated starting point — check the loss curve and increase if it hasn't plateaued.

## Preprocessing (GPU-only, optional)

`experimental/preprocess.py` regenerates `preprocessed/<base>_preprocessed/` from raw `.inp`/`.crs` files via `includes/preprocessor.py` (`cupy`/`numba`, needs `environment.yml`). Edit `base_name` at the top of the script to target a different case. Not needed to run the four bundled datasets.

## Known limitations
- `includes/mech.py`'s mechanics stride (every 10th thermal step) and `includes/utils.py:save_vtk`'s VTK-timing calc are separately hardcoded but currently consistent (`dt=0.01 × stride 10 = 0.1`) — changing one without the other will desync export timestamps.
- Memory grows as `O(N_timesteps × N_mesh)` (gradients unroll through time, no checkpointing), bounding case size per GPU.
- `includes/preprocessor.py`'s `write_parameters` still writes a legacy IN718 material template into the `.k` file; the JAX pipeline ignores it and reads Ti-6Al-4V from `materials/TI64_*_Debroy.txt` directly.
