import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

output_dir = "./opt_MWE_reg"  # Change if needed
save_dir = output_dir  # Save animation in the same directory

# Choose which to animate: 'params_' or 'control_'
prefix = "control_"  # or "control_"

# Find and sort all matching PNG files
files = [f for f in os.listdir(output_dir) if f.startswith(prefix) and f.endswith('.png')]
files.sort()

# Load images
images = [Image.open(os.path.join(output_dir, f)) for f in files]

# Create animation and save as MP4
fig = plt.figure(figsize=(8, 4))
plt.axis('off')
ims = [[plt.imshow(img, animated=True)] for img in images]

# ani = animation.ArtistAnimation(fig, ims, interval=300, blit=True, repeat_delay=1000)
# mp4_path = os.path.join(save_dir, f"{prefix}animation.mp4")
# ani.save(mp4_path, writer='ffmpeg')
# print(f"Saved animation to {mp4_path}")

# Optionally, save as GIF (for PowerPoint compatibility)
gif_path = os.path.join(save_dir, f"{prefix}animation.gif")
images[0].save(
    gif_path,
    save_all=True,
    append_images=images[1:],
    duration=300,
    loop=0
)
print(f"Saved animation to {gif_path}")