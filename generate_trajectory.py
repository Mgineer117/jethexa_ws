#!/usr/bin/env python3
"""
JET-HEXA TRAJECTORY ANALYZER
----------------------------
Functionality:
1. Selects the most recently modified .npz file.
2. Extracts the 'states' and 'controls' arrays from the archive.
3. Parses the data according to the Data Collector format:
    STATES (24-dim):
    - Indices [0:3] : Base Position (x, y, z)
    - Indices [3:6] : Base Orientation (x, y, z / Euler)
    - Indices [6:24]: Joint Positions (18 DOF)

    CONTROLS (18-dim):
    - Indices [0:18]: Joint Velocities / Position Deltas
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def analyze_hexapod_trajectories(directory="."):
    # 1. Find all .npz files in the directory
    files = glob.glob(os.path.join(directory, "*.npz"))
    if not files:
        print("No .npz files found in directory!")
        return

    # Sort by modification time to get the latest recording
    latest_file = max(files, key=os.path.getmtime)
    print(f"--- Analyzing Latest Recording: {latest_file} ---")

    # 2. Load the archive and extract the arrays
    archive = np.load(latest_file)

    # Verify the keys inside the archive
    print(f"Archive contains keys: {archive.files}")

    states_data = archive["states"]
    controls_data = archive["controls"]

    timesteps = np.arange(states_data.shape[0]) / 10.0  # Assuming 10Hz recording

    # 3. Create a multi-plot figure
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Trajectory Analysis: {os.path.basename(latest_file)}", fontsize=16)

    # --- Plot 1: Base Orientation (Indices 3:6) ---
    labels = ["x / roll", "y / pitch", "z / yaw"]
    for i in range(3):
        axs[0].plot(
            timesteps, states_data[:, i + 3], label=f"Ori {labels[i]}", linewidth=2
        )
    axs[0].set_ylabel("Angle (rad)")
    axs[0].set_title("Torso Orientation (From Qualisys)")
    axs[0].legend(loc="upper right", ncol=3, fontsize="small")
    axs[0].grid(True, alpha=0.3)

    # --- Plot 2: Joint Positions (Indices 6:24 - Showing first 6 joints) ---
    # We plot index 6 through 11 (First 6 joints)
    for i in range(6, 12):
        axs[1].plot(timesteps, states_data[:, i], label=f"Joint {i-5}")
    axs[1].set_ylabel("Position (rad)")
    axs[1].set_title("Joint Positions (Legs 1-2)")
    axs[1].legend(loc="upper right", ncol=3, fontsize="x-small")
    axs[1].grid(True, alpha=0.3)

    # --- Plot 3: Joint Controls/Velocities (Indices 0:18 in Controls Array - Showing first 6) ---
    # We plot index 0 through 5 from the controls_data array
    for i in range(0, 6):
        axs[2].plot(timesteps, controls_data[:, i], label=f"Ctrl/Vel {i+1}")
    axs[2].set_ylabel("Delta Position / Vel")
    axs[2].set_xlabel("Time (seconds)")
    axs[2].set_title("Joint Controls (Delta from previous step)")
    axs[2].legend(loc="upper right", ncol=3, fontsize="x-small")
    axs[2].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    print("Plotting complete.")
    plt.show()


if __name__ == "__main__":
    # Ensure this directory matches the output_dir in your collector
    analyze_hexapod_trajectories("hexapod_data")
