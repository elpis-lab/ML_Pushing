import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def plot_deltas():
    # 1. Set up Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('obj_name', type=str, help='Name of the object (e.g., mustard_bottle_flipped)')
    parser.add_argument('n_data', type=int, help='Number of data points (e.g., 1000)')
    args = parser.parse_args()

    # 2. Construct file paths
    x_path = f"data/x_{args.obj_name}_{args.n_data}.npy"
    y_path = f"data/y_{args.obj_name}_{args.n_data}.npy"

    # 3. Load the data
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        print(f"Error: Data files not found at {x_path} or {y_path}")
        return

    x_data = np.load(x_path)    # Shape: (N, 8)
    y_data = np.load(y_path)    # Shape: (N, 3)

    # 4. Extract features from x_data
    # Columns: [x, y, z, nx, ny, nz, angle]
    contact_points = x_data[:, :3]      # (N, 3)
    contact_normals = x_data[:, 3:6]    # (N, 3)
    angles = x_data[:, 6]               # (N,)
    curvatures = x_data[:, 7]           # (N,)
    
    # Deltas (N, 3) -> dx, dy, dtheta
    deltas = y_data

    # 5. Create plots
    fig, axs = plt.subplots(4, 1, figsize=(10, 15))

    # Plot 1: Angle vs Delta X
    axs[0].scatter(angles, deltas[:, 0], alpha=0.4, color='blue', s=10)
    axs[0].set_title(f'Push Angle vs $\Delta x$ ({args.obj_name})')
    axs[0].set_ylabel('$\Delta x$ (m)')
    axs[0].grid(True)

    # Plot 2: Angle vs Delta Y
    axs[1].scatter(angles, deltas[:, 1], alpha=0.4, color='green', s=10)
    axs[1].set_title(f'Push Angle vs $\Delta y$ ({args.obj_name})')
    axs[1].set_ylabel('$\Delta y$ (m)')
    axs[1].grid(True)

    # Plot 3: Angle vs Delta Theta
    axs[2].scatter(angles, deltas[:, 2], alpha=0.4, color='red', s=10)
    axs[2].set_title(f'Push Angle vs $\Delta \\theta$ ({args.obj_name})')
    # axs[2].set_xlabel('Push Angle (rad)')
    axs[2].set_ylabel('$\Delta \\theta$ (rad)')
    axs[2].grid(True)

    # Plot 4: Angle vs curvature Theta
    axs[3].scatter(angles, curvatures, alpha=0.4, color='black', s=10)
    axs[3].set_title(f'Push Angle vs Curvature({args.obj_name})')
    axs[3].set_xlabel('Push Angle (rad)')
    axs[3].set_ylabel('Curvature')
    axs[3].grid(True)

    plt.tight_layout()
    save_path = f'data/plot_{args.obj_name}_{args.n_data}.png'
    plt.savefig(save_path)
    print(f"Plots saved to '{save_path}'.")

if __name__ == "__main__":
    plot_deltas()