#!/usr/bin/env python3
"""
JET-HEXA TRAJECTORY ANALYZER
----------------------------
Functionality:
1. Selects the most recently modified .npy file.
2. Parses the 40-dimensional state vector (QUAT FIRST):
    - Indices [0:4]: Base Orientation (IMU Quaternions x, y, z, w)
    - Indices [4:22]: Joint Positions (18 DOF)
    - Indices [22:40]: Joint Velocities (Finite Difference)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob


def analyze_hexapod_trajectories(directory="."):
    # 1. Find all .npy files in the directory
    files = glob.glob(os.path.join(directory, "*.npy"))
    if not files:
        print("No .npy files found in directory!")
        return

    # Sort by modification time to get the latest recording
    latest_file = max(files, key=os.path.getmtime)
    print(f"--- Analyzing Latest Recording (QUAT FIRST): {latest_file} ---")

    # 2. Load the data
    data = np.load(latest_file)
    timesteps = np.arange(data.shape[0]) / 10.0  # 10Hz

    # 3. Create a multi-plot figure
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Trajectory Analysis: {os.path.basename(latest_file)}", fontsize=16)

    # --- Plot 1: IMU Orientation (Indices 0:4) ---
    labels = ["x", "y", "z", "w"]
    for i in range(4):
        axs[0].plot(timesteps, data[:, i], label=f"Quat {labels[i]}", linewidth=2)
    axs[0].set_ylabel("Quaternion Value")
    axs[0].set_title("Torso Orientation (IMU Filtered)")
    axs[0].legend(loc="upper right", ncol=4, fontsize="small")
    axs[0].grid(True, alpha=0.3)

    # --- Plot 2: Joint Positions (Indices 4:22 - Showing first 6 joints) ---
    # We plot index 4 through 9 (First 6 joints)
    for i in range(4, 10):
        axs[1].plot(timesteps, data[:, i], label=f"Joint {i-3}")
    axs[1].set_ylabel("Position (rad)")
    axs[1].set_title("Joint Positions (Legs 1-2)")
    axs[1].legend(loc="upper right", ncol=3, fontsize="x-small")
    axs[1].grid(True, alpha=0.3)

    # --- Plot 3: Joint Velocities (Indices 22:40 - Showing first 6 joints) ---
    # We plot index 22 through 27 (First 6 velocities)
    for i in range(22, 28):
        axs[2].plot(timesteps, data[:, i], label=f"Vel {i-21}")
    axs[2].set_ylabel("Velocity (rad/s)")
    axs[2].set_xlabel("Time (seconds)")
    axs[2].set_title("Joint Velocities (Finite Difference)")
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("Plotting complete.")
    plt.show()


if __name__ == "__main__":
    analyze_hexapod_trajectories("hexapod_data/")
