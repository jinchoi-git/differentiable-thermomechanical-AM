import os
import numpy as np
import matplotlib.pyplot as plt
import re

output_dir = "./opt_MWE_reg_nosmooth"  # Change if your output directory is different
plot_dir = os.path.join(output_dir, "control_plots")
os.makedirs(plot_dir, exist_ok=True)

# Find all control_####.npy files
files = [f for f in os.listdir(output_dir) if f.startswith('control_') and f.endswith('.npy')]
files.sort()  # Sorts by string, which works for zero-padded numbers

# Plot and save every 10th control file
for i, fname in enumerate(files):
    if i % 1 == 0:
        control = np.load(os.path.join(output_dir, fname))
        plt.figure(figsize=(8, 4))
        plt.plot(control[:500], marker='o', linestyle='-')
        plt.title(f'Control at Iteration {fname[8:12]}')
        plt.xlabel('Time Step')
        plt.ylabel('Control Value')
        plt.tight_layout()
        save_path = os.path.join(plot_dir, f"control_plot_{fname[8:12]}.png")
        plt.savefig(save_path)
        plt.close()
        print(f"Saved {save_path}")