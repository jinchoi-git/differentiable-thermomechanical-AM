import os
import numpy as np
import matplotlib.pyplot as plt
glob_dir = "/home/jyc3887/differentiable-thermomechanical-AM"  # Base directory containing minimize_stress_* folders
output_dir = "./analysis_combined"

os.makedirs(output_dir, exist_ok=True)

def plot_all_controls_and_losses(parent_dir="."):
    import glob
    
    dirs = sorted(glob.glob(os.path.join(parent_dir, "minimize_stress_*")))
    print(f"Found {len(dirs)} directories.")

    # Plot final control vectors
    plt.figure(figsize=(8, 5), dpi=300)

    for dir_path in dirs:
        print(f"Processing directory: {dir_path}")

        # Find latest control file
        control_files = [f for f in os.listdir(dir_path) if f.startswith('control_') and f.endswith('.npy')]
        if control_files:
            control_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))
            final_control = np.load(os.path.join(dir_path, control_files[-1]))[0:500]
            print(final_control.shape)
            plt.plot(final_control, label=os.path.basename(dir_path))
        else:
            print(f"⚠ No control files found in {dir_path}")

    plt.xlabel("Timestep")
    plt.ylabel("Normalized Control")
    plt.title("Final Control Vectors")
    plt.legend(fontsize="small", loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "final_controls_all.png"))
    plt.show()

    # Plot loss histories side by side (log scale)
    plt.figure(figsize=(8, 5), dpi=300)
    
    for dir_path in dirs:
        loss_files = [f for f in os.listdir(dir_path) if f.startswith('loss_') and f.endswith('.npy')]
        if loss_files:
            loss_files.sort(key=lambda f: int(f.split('_')[1].split('.')[0]))
            loss_history = np.load(os.path.join(dir_path, loss_files[-1]))
            plt.plot(loss_history, label=os.path.basename(dir_path))
        else:
            print(f"⚠ No loss files found in {dir_path}")

    plt.yscale('log')  # Set y-axis to log scale
    plt.xlabel("Iteration")
    plt.ylabel("Loss (log scale)")
    plt.title("Loss History Across Runs")
    plt.legend(fontsize="small", loc="upper right")
    plt.grid(True, which="both", linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_histories_all.png"))
    plt.show()

if __name__ == "__main__":
    plot_all_controls_and_losses(glob_dir)
