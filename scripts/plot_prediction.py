import numpy as np
import matplotlib.pyplot as plt
import os

def plot_prediction(
    pred_path  = "data/pred.npy",
    x_path     = "data/x_mustard_bottle_flipped_1000.npy",
    save_path  = "data/plot_prediction.png",
):
    for p in (pred_path, x_path):
        if not os.path.exists(p):
            print(f"Error: file not found → {p}")
            return

    pred = np.load(pred_path)   # (N, 3): dx, dy, dtheta
    x    = np.load(x_path)      # (N, 3): px, py, angle

    dx     = pred[:, 0]
    dy     = pred[:, 1]
    dtheta = pred[:, 2]
    angles = x[:, 2]

    # sort by push angle for cleaner plots
    sort_idx = np.argsort(angles)
    angles   = angles[sort_idx]
    dx       = dx[sort_idx]
    dy       = dy[sort_idx]
    dtheta   = dtheta[sort_idx]

    fig, axs = plt.subplots(3, 1, figsize=(10, 9))

    axs[0].scatter(angles, dx, alpha=0.4, color='blue', s=10)
    axs[0].set_title('Push Angle vs $\\Delta x$')
    axs[0].set_ylabel('$\\Delta x$ (m)')
    axs[0].grid(True)

    axs[1].scatter(angles, dy, alpha=0.4, color='green', s=10)
    axs[1].set_title('Push Angle vs $\\Delta y$')
    axs[1].set_ylabel('$\\Delta y$ (m)')
    axs[1].grid(True)

    axs[2].scatter(angles, dtheta, alpha=0.4, color='red', s=10)
    axs[2].set_title('Push Angle vs $\\Delta\\theta$')
    axs[2].set_xlabel('Push Angle (rad)')
    axs[2].set_ylabel('$\\Delta\\theta$ (rad)')
    axs[2].grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Saved → {save_path}")

if __name__ == "__main__":
    plot_prediction()