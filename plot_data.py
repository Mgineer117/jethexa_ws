#!/usr/bin/env python3
"""
JET-HEXA TRAJECTORY ANALYZER (MULTI-FILE)
-----------------------------------------
Functionality:
1. Loads all .npz files in the specified directory.
2. Performs sanity checks (removes files with NaNs or length < 300).
3. Extracts the 'states' array from valid archives.
4. Plots all valid trajectories overlaid across 24 independent subplots.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def analyze_hexapod_trajectories(directory="."):
    # 1. Find all .npz files
    files = glob.glob(os.path.join(directory, "*.npz"))
    if not files:
        print(f"No .npz files found in directory: {directory}")
        return

    valid_files = []
    data_dict = {}

    print(f"Found {len(files)} files. Running sanity checks...")

    # 2. Sanity Checks
    for f in files:
        try:
            archive = np.load(f)
            states = archive["states"]

            # Check for NaNs
            if np.isnan(states).any():
                print(f"  [REMOVED] {os.path.basename(f)}: Contains NaN values.")
                continue

            # Check minimum length
            if states.shape[0] < 300:
                print(
                    f"  [REMOVED] {os.path.basename(f)}: Length ({states.shape[0]}) is less than 300 steps."
                )
                continue

            valid_files.append(f)
            data_dict[f] = states

        except Exception as e:
            print(f"  [REMOVED] {os.path.basename(f)}: Failed to load ({e}).")

    if not valid_files:
        print("No valid files remain after sanity checks. Exiting.")
        return

    print(f"--- Plotting {len(valid_files)} valid trajectories ---")

    # 3. Create a 8x3 multi-plot figure (24 dimensions)
    # Row 0: Pos (x,y,z), Row 1: Ori (r,p,y), Rows 2-7: Joints (6 legs x 3 DOFs)
    fig, axs = plt.subplots(8, 3, figsize=(16, 20), sharex=True)
    fig.suptitle("Hexapod Trajectory Analysis (All Valid Runs)", fontsize=18, y=0.98)

    axs_flat = axs.flatten()

    # Define labels for the 24 subplots
    labels = [
        "Base Pos X",
        "Base Pos Y",
        "Base Pos Z",
        "Base Roll",
        "Base Pitch",
        "Base Yaw",
    ]
    labels += [f"Joint {i}" for i in range(1, 19)]

    # 4. Plot all valid data
    for f in valid_files:
        states_data = data_dict[f]
        timesteps = np.arange(states_data.shape[0]) / 10.0  # Assuming 10Hz recording
        fname = os.path.basename(f)

        for i in range(24):
            axs_flat[i].plot(
                timesteps, states_data[:, i], label=fname, linewidth=1.5, alpha=0.8
            )

    # 5. Formatting the subplots
    for i in range(24):
        axs_flat[i].set_title(labels[i], fontsize=10)
        axs_flat[i].grid(True, alpha=0.3)

        # Add Y-axis labels based on data type
        if i < 3:
            axs_flat[i].set_ylabel("m", fontsize=8)
        else:
            axs_flat[i].set_ylabel("rad", fontsize=8)

    # Add X-axis labels to the bottom row only
    for i in range(21, 24):
        axs_flat[i].set_xlabel("Time (seconds)")

    # 6. Global Legend Layout
    # Extract handles and labels from the first subplot to create a single master legend
    handles, leg_labels = axs_flat[0].get_legend_handles_labels()

    # Adjust layout to make room for the legend on the right side
    plt.tight_layout(rect=[0, 0, 0.85, 0.97])
    fig.legend(
        handles,
        leg_labels,
        loc="center right",
        bbox_to_anchor=(0.99, 0.5),
        fontsize="small",
        title="File Names",
    )

    print("Plotting complete. Rendering window...")
    plt.show()


if __name__ == "__main__":
    # Ensure this directory matches the output_dir in your collector
    analyze_hexapod_trajectories("hexapod_data")
