# --- Visualization & Animation helpers ---------------------------------------
from typing import List, Optional, Tuple
import os, re, glob, warnings, datetime
import numpy as np
from PIL import Image
import imageio.v3 as iio
import jax
import jax.numpy as jnp
import pyvista as pv
import vtk
import matplotlib.pyplot as plt
from typing import Sequence
from .mech import transformation

_num_re = re.compile(r"(\d+)")

def make_run_dir(base_dir: str, mode: str, tag: str | None = None, timestamp: bool = False) -> str:
    """
    Create and return a subdirectory like:
      base_dir/mode/[YYYYmmdd-HHMMSS_][tag]
    If both timestamp=False and tag=None, returns base_dir/mode.
    """
    parts = [base_dir, mode]
    if timestamp:
        parts.append(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    if tag:
        parts.append(tag)
    run_dir = os.path.join(*parts)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def latest_control_under(work_root: str, mode: str) -> tuple[str, str]:
    """
    Returns (ctrl_path, run_dir) for the newest control_*.npy found under
    work_root/mode/**. 'run_dir' is the parent folder of that control file.
    """
    search_root = os.path.join(work_root, mode)
    # recurse for any timestamp/tag subfolders
    paths = glob.glob(os.path.join(search_root, "**", "control_*.npy"), recursive=True)
    if not paths:
        raise FileNotFoundError(f"No control_*.npy found under {search_root}")
    # newest by modification time
    ctrl_path = max(paths, key=os.path.getmtime)
    run_dir = os.path.dirname(ctrl_path)
    return ctrl_path, run_dir

def save_vtk(T_seq, S_seq, U_seq, elements, Bip_ele, nodes, element_birth, node_birth, dt, run_dir="./vtk_out", keyword="forward"):
    T_total = S_seq.shape[0]

    def get_detJacs(element):
        nodes_pos = nodes[element]
        Jac = jnp.matmul(Bip_ele, nodes_pos)
        ele_detJac_ = jnp.linalg.det(Jac)
        return ele_detJac_
    
    ele_detJac = jax.vmap(get_detJacs)(elements)
    
    for t in range(0, T_total):
        dt = 0.1 
        current_time = t * dt
        filename = os.path.join(run_dir, f"{keyword}_{t:04d}.vtk")
        
        S = S_seq[t]
        U = U_seq[t]
        T = T_seq[int(t*10)]   
        
        # Recompute activation masks at this time
        active_element_inds = np.array(element_birth <= current_time)
        active_node_inds = np.array(node_birth <= current_time)
        n_e_save = int(np.sum(active_element_inds))
        n_n_save = int(np.sum(active_node_inds))
        
        active_elements_list = elements[active_element_inds].tolist()
        active_cells = np.array([item for sublist in active_elements_list for item in [8] + sublist])
        active_cell_type = np.array([vtk.VTK_HEXAHEDRON] * len(active_elements_list))
        
        points = np.array(nodes[0:n_n_save] + U[0:n_n_save])
        # points = np.array(nodes[0:n_n_save])
        
        Sv = transformation(np.sqrt(1/2 * ((S[0:n_e_save,:,0] - S[0:n_e_save,:,1])**2 + 
                                        (S[0:n_e_save,:,1] - S[0:n_e_save,:,2])**2 + 
                                        (S[0:n_e_save,:,2] - S[0:n_e_save,:,0])**2 + 
                                        6 * (S[0:n_e_save,:,3]**2 + S[0:n_e_save,:,4]**2 + S[0:n_e_save,:,5]**2))), 
                        elements[0:n_e_save], ele_detJac[0:n_e_save], n_n_save)
        S11 = transformation(S[0:n_e_save,:,0], elements[0:n_e_save], ele_detJac[0:n_e_save], n_n_save)
        S22 = transformation(S[0:n_e_save,:,1], elements[0:n_e_save], ele_detJac[0:n_e_save], n_n_save)
        S33 = transformation(S[0:n_e_save,:,2], elements[0:n_e_save], ele_detJac[0:n_e_save], n_n_save)
        S12 = transformation(S[0:n_e_save,:,3], elements[0:n_e_save], ele_detJac[0:n_e_save], n_n_save)
        S23 = transformation(S[0:n_e_save,:,4], elements[0:n_e_save], ele_detJac[0:n_e_save], n_n_save)
        S13 = transformation(S[0:n_e_save,:,5], elements[0:n_e_save], ele_detJac[0:n_e_save], n_n_save)
    
        # Using pyvista for vtk
        active_grid = pv.UnstructuredGrid(active_cells, active_cell_type, points)
        # active_grid.point_data['temp'] = np.clip(np.array(T[0:n_n_save]), 300, 2300)
        active_grid.point_data['temp'] = np.array(T[0:n_n_save])
        active_grid.point_data['S_von'] = np.array(Sv)
        active_grid.point_data['S11'] = np.array(S11)
        active_grid.point_data['S22'] = np.array(S22)
        active_grid.point_data['S33'] = np.array(S33)
        active_grid.point_data['S12'] = np.array(S12)
        active_grid.point_data['S23'] = np.array(S23)
        active_grid.point_data['S13'] = np.array(S13)
        active_grid.point_data['U1'] = np.array(U[0:n_n_save, 0])
        active_grid.point_data['U2'] = np.array(U[0:n_n_save, 1])
        active_grid.point_data['U3'] = np.array(U[0:n_n_save, 2])
        active_grid.save(filename)

def find_latest(prefix, run_dir):
    files = [f for f in os.listdir(run_dir) if f.startswith(prefix) and f.endswith('.npy')]
    print(f"Found {len(files)} '{prefix}_*.npy' files")
    if not files:
        raise FileNotFoundError(f"No files with prefix '{prefix}' in {run_dir}")
    iters = [int(f.split('_')[1].split('.')[0]) for f in files]
    latest_iter = max(iters)
    latest_file = f"{prefix}_{latest_iter:04d}.npy"
    print(f"→ Latest {prefix} file: {latest_file} (iter {latest_iter})")
    return os.path.join(run_dir, latest_file), latest_iter

def _numeric_key(path: str):
    """Sorts ..._0000.png, ..._0001.png, ..._0010.png in numeric order."""
    m = list(_num_re.finditer(os.path.basename(path)))
    return int(m[-1].group(1)) if m else path

def _collect_frames(run_dir: str, pattern: str) -> List[str]:
    paths = glob.glob(os.path.join(run_dir, pattern))
    paths.sort(key=_numeric_key)
    return paths

def _ensure_even_size(img: Image.Image) -> Image.Image:
    """Pad by 1 px if needed; some encoders prefer even width/height."""
    w, h = img.size
    pad_w = w % 2
    pad_h = h % 2
    if pad_w or pad_h:
        new = Image.new(img.mode, (w + pad_w, h + pad_h), color="white")
        new.paste(img, (0, 0))
        return new
    return img

def _write_gif(frames: List[Image.Image], out_path: str, fps: int = 2, loop: int = 0):
    if not frames:
        return
    duration_ms = int(1000 / max(1, fps))  # GIF uses per-frame duration in ms
    frames[0].save(
        out_path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,     # safer for many frames
        duration=duration_ms,
        loop=loop,
        disposal=2,
    )

def _write_mp4(frames: List[Image.Image], out_path: str, fps: int = 2, quality: int = 8):
    if not frames:
        return
    nd = [np.array(_ensure_even_size(im).convert("RGB")) for im in frames]
    try:
        iio.imwrite(out_path, nd, fps=fps, codec="libx264", quality=quality)
    except Exception as e:
        warnings.warn(f"MP4 export failed ({e}). Is ffmpeg available?). Skipping MP4.")

def make_animation_from_pattern(
    run_dir: str,
    pattern: str,
    out_stem: str,
    fps: int = 2,
    resize_to: Optional[Tuple[int, int]] = None,
    make_gif: bool = True,
    make_mp4: bool = True,
) -> Tuple[Optional[str], Optional[str]]:
    """
    Convert a sequence of PNGs in run_dir matching `pattern` into GIF/MP4.
    Returns (gif_path, mp4_path).
    """
    paths = _collect_frames(run_dir, pattern)
    if not paths:
        print(f"[animate] No frames found for pattern '{pattern}' in {run_dir}")
        return (None, None)

    frames: List[Image.Image] = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        if resize_to is not None:
            im = im.resize(resize_to, Image.LANCZOS)
        frames.append(im)

    gif_path = os.path.join(run_dir, f"{out_stem}.gif") if make_gif else None
    mp4_path = os.path.join(run_dir, f"{out_stem}.mp4") if make_mp4 else None

    if make_gif and gif_path:
        _write_gif(frames, gif_path, fps=fps, loop=0)
        print(f"[animate] Wrote {gif_path}")

    if make_mp4 and mp4_path:
        _write_mp4(frames, mp4_path, fps=fps, quality=8)
        if os.path.exists(mp4_path):
            print(f"[animate] Wrote {mp4_path}")

    # Close image file handles
    for im in frames:
        try:
            im.close()
        except:
            pass

    return (gif_path if make_gif else None, mp4_path if make_mp4 else None)

def make_iteration_dashboard(
    run_dir: str,
    left_pattern="params_plot_*.png",
    mid_pattern="control_plot_*.png",
    right_pattern="loss_history_plot_*.png",
    out_stem="dashboard",
    fps=2,
    width_per_panel=800,
    pad=10,
    make_gif=True,
    make_mp4=True,
):
    """
    For each iteration, horizontally concatenate params/control/loss plots into
    a single 'dashboard' frame, then animate.
    """
    lefts  = _collect_frames(run_dir, left_pattern)
    mids   = _collect_frames(run_dir, mid_pattern)
    rights = _collect_frames(run_dir, right_pattern)
    if not (lefts and mids and rights):
        print("[dashboard] Not all panels found; skipping dashboard animation.")
        return (None, None)

    def _numeric_index_map(paths):
        return {_numeric_key(p): p for p in paths}

    L = _numeric_index_map(lefts)
    M = _numeric_index_map(mids)
    R = _numeric_index_map(rights)
    common = sorted(set(L.keys()) & set(M.keys()) & set(R.keys()))
    if not common:
        print("[dashboard] No common iterations across panels.")
        return (None, None)

    frames = []
    target_h = None
    for k in common:
        imgs = [Image.open(L[k]).convert("RGB"),
                Image.open(M[k]).convert("RGB"),
                Image.open(R[k]).convert("RGB")]
        if target_h is None:
            target_h = max(im.height for im in imgs)
        resized = [
            im.resize((int(im.width * (target_h / im.height)), target_h), Image.LANCZOS)
            for im in imgs
        ]
        total_w = sum(im.width for im in resized) + 2 * pad
        canvas = Image.new("RGB", (total_w + 2 * pad, target_h + 2 * pad), "white")
        x = pad
        for im in resized:
            canvas.paste(im, (x, pad))
            x += im.width + pad

        final_w = 3 * width_per_panel + 4 * pad
        canvas = canvas.resize(
            (final_w, int(canvas.height * (final_w / canvas.width))), Image.LANCZOS
        )
        frames.append(canvas)

    gif_path = os.path.join(run_dir, f"{out_stem}.gif") if make_gif else None
    mp4_path = os.path.join(run_dir, f"{out_stem}.mp4") if make_mp4 else None

    if make_gif and gif_path:
        _write_gif(frames, gif_path, fps=fps, loop=0)
        print(f"[dashboard] Wrote {gif_path}")
    if make_mp4 and mp4_path:
        _write_mp4(frames, mp4_path, fps=fps, quality=8)
        if os.path.exists(mp4_path):
            print(f"[dashboard] Wrote {mp4_path}")

    for im in frames:
        try:
            im.close()
        except:
            pass

    return (gif_path if make_gif else None, mp4_path if make_mp4 else None)
# -------------------------------------------------------------------------------

def save_iter_artifacts(
    iteration: int,
    params_np: np.ndarray,
    control_np: np.ndarray,
    loss_history: Sequence[float],
    run_dir: str,
    power_on_steps: int,
) -> None:
    """
    Save params/control/loss arrays and 3 quick plots for a given iteration.
    """
    os.makedirs(run_dir, exist_ok=True)

    # Arrays
    np.save(os.path.join(run_dir, f"params_{iteration:04d}.npy"), np.array(params_np))
    np.save(os.path.join(run_dir, f"control_{iteration:04d}.npy"), np.array(control_np))
    np.save(os.path.join(run_dir, f"loss_{iteration:04d}.npy"),    np.array(loss_history))

    # Plots
    plt.figure(figsize=(8, 4))
    plt.plot(np.array(params_np), marker="o", linestyle="-")
    plt.title(f"Params at Iteration {iteration}")
    plt.xlabel("Parameter Index"); plt.ylabel("Parameter Value")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"params_plot_{iteration:04d}.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(np.array(control_np)[:power_on_steps], marker="o", linestyle="-")
    plt.title(f"Control at Iteration {iteration}")
    plt.xlabel("Time Step"); plt.ylabel("Control Value")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"control_plot_{iteration:04d}.png"))
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(np.array(loss_history), marker="o", linestyle="-")
    plt.title("Loss history")
    plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, f"loss_history_plot_{iteration:04d}.png"))
    plt.close()
# -------------------------------------------------------------------------------