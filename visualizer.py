import os
import numpy as np
import matplotlib.pyplot as plt

# Path to your training outputs
dir_path = './implicit_debug'
print(f"Looking in directory: {dir_path}")

# Assume control is shape (n_steps,)
n_steps = 500

# Generate target sine wave baseline
x = np.linspace(0, 2 * np.pi, n_steps)
baseline = 1.0 + 0.25 * np.sin(x)  # oscillates between 0.75 and 1.25


# Helper to find the highest iteration number from filenames like 'loss_0005.npy'
def find_latest(prefix):
    files = [f for f in os.listdir(dir_path) if f.startswith(prefix) and f.endswith('.npy')]
    print(f"Found {len(files)} '{prefix}_*.npy' files")
    if not files:
        raise FileNotFoundError(f"No files with prefix '{prefix}' in {dir_path}")
    iters = [int(f.split('_')[1].split('.')[0]) for f in files]
    latest_iter = max(iters)
    latest_file = f"{prefix}_{latest_iter:04d}.npy"
    print(f"→ Latest {prefix} file: {latest_file} (iter {latest_iter})")
    return os.path.join(dir_path, latest_file), latest_iter

# Load latest loss history
loss_path, loss_iter = find_latest('loss')
print(f"Loading loss history from: {loss_path}")
loss = np.load(loss_path)
print(f"Loaded loss array of shape {loss.shape}")

# Plot the loss history
print("Plotting loss history...")
plt.figure(figsize=(6,4))
plt.plot(loss, marker='o', markersize=3, linewidth=1)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title(f'Loss History (up to iter {loss_iter})')
plt.grid(True)
plt.tight_layout()
# plt.show()
print("Done plotting loss.")

# Collect all control files and sort by iteration number
control_files = [f for f in os.listdir(dir_path) if f.startswith('control_') and f.endswith('.npy')]
control_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))

# Load all control arrays
controls = [np.load(os.path.join(dir_path, f)) for f in control_files]
iters = [int(f.split('_')[1].split('.')[0]) for f in control_files]

# Plot every 10th control
plt.figure(figsize=(8, 4))
for i, (ctrl, it) in enumerate(zip(controls, iters)):
    if i % 10 == 0:
        plt.plot(ctrl[0:500], label=f'Iter {it}', linewidth=1)
    if i == len(controls) - 1:
       plt.plot(ctrl[0:500], label=f'Iter {len(controls)-1}', linewidth=1)

plt.plot(baseline, '--', color='gray', label='Target Baseline')
plt.xlabel('Timestep')
plt.ylabel('Normalized Control')
plt.title('Control History')
plt.legend(ncol=3, fontsize='small')
plt.grid(True)
plt.tight_layout()

plt.show()

