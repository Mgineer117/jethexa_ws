#!/usr/bin/env python3
"""
JET-HEXA TRAJECTORY & CONTROL ANALYZER (MULTI-FILE)
---------------------------------------------------
Functionality:
1. Loads all .npz files in the specified directory.
2. Performs sanity checks (removes files with NaNs or length < 300).
3. Extracts both 'states' and 'controls' arrays.
4. Calculates and prints global min/max bounds for all elements,
   including exact pi/n fractional mapping for angles.
5. Plots states overlaid across a 24-subplot figure.
6. Plots controls overlaid across an 18-subplot figure.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def exact_pi_fraction(val):
    """
    Helper function to strictly convert a decimal radian value
    into its exact 'pi/n' fractional representation.
    """
    if val == 0:
        return "0"

    # Calculate the exact denominator
    n = np.pi / abs(val)

    # Round to the nearest whole integer
    n_int = int(round(n))

    sign = "-" if val < 0 else " "
    if n_int == 1:
        return f"{sign}pi"
    return f"{sign}pi/{n_int}"


def analyze_hexapod_trajectories(directory="."):
    # 1. Find all .npz files
    files = glob.glob(os.path.join(directory, "*.npz"))
    if not files:
        print(f"No .npz files found in directory: {directory}")
        return

    valid_files = []
    states_dict = {}
    controls_dict = {}

    # Initialize variables to track global bounds
    global_states_min = None
    global_states_max = None
    global_controls_min = None
    global_controls_max = None

    print(f"Found {len(files)} files. Running sanity checks...")

    # 2. Sanity Checks & Bound Tracking
    for f in files:
        try:
            archive = np.load(f)
            states = archive["states"]
            controls = archive["controls"]

            # Check for NaNs
            if np.isnan(states).any() or np.isnan(controls).any():
                print(f"  [REMOVED] {os.path.basename(f)}: Contains NaN values.")
                continue

            # Check minimum length
            if states.shape[0] < 300:
                print(
                    f"  [REMOVED] {os.path.basename(f)}: Length ({states.shape[0]}) is less than 300 steps."
                )
                continue

            valid_files.append(f)
            states_dict[f] = states
            controls_dict[f] = controls

            # Update global min/max bounds dynamically
            if global_states_min is None:
                global_states_min = np.min(states, axis=0)
                global_states_max = np.max(states, axis=0)
                global_controls_min = np.min(controls, axis=0)
                global_controls_max = np.max(controls, axis=0)
            else:
                global_states_min = np.minimum(
                    global_states_min, np.min(states, axis=0)
                )
                global_states_max = np.maximum(
                    global_states_max, np.max(states, axis=0)
                )
                global_controls_min = np.minimum(
                    global_controls_min, np.min(controls, axis=0)
                )
                global_controls_max = np.maximum(
                    global_controls_max, np.max(controls, axis=0)
                )

        except Exception as e:
            print(f"  [REMOVED] {os.path.basename(f)}: Failed to load ({e}).")

    if not valid_files:
        print("No valid files remain after sanity checks. Exiting.")
        return

    # Define labels early so we can use them for printing
    labels_x = [
        "Base Pos X",
        "Base Pos Y",
        "Base Pos Z",
        "Base Roll",
        "Base Pitch",
        "Base Yaw",
    ] + [f"State Joint {i}" for i in range(1, 19)]

    labels_u = [f"Control Joint {i}" for i in range(1, 19)]

    # ==========================================
    # 3. PRINT EMPIRICAL BOUNDS
    # ==========================================
    print("\n" + "=" * 80)
    print("EMPIRICAL BOUNDS (Across all valid files)")
    print("=" * 80)

    print("\n--- STATE BOUNDS ---")
    for i in range(len(labels_x)):
        min_val = global_states_min[i]
        max_val = global_states_max[i]

        # If the state is an angle (indices 3 through 23), print the pi fraction
        if i >= 3:
            min_pi = exact_pi_fraction(min_val)
            max_pi = exact_pi_fraction(max_val)
            print(
                f"{labels_x[i]:<15}: Min = {min_val:>8.4f} ({min_pi:>8}), Max = {max_val:>8.4f} ({max_pi:>8})"
            )
        else:
            print(
                f"{labels_x[i]:<15}: Min = {min_val:>8.4f}           , Max = {max_val:>8.4f}"
            )

    print("\n--- CONTROL BOUNDS ---")
    for i in range(len(labels_u)):
        print(
            f"{labels_u[i]:<17}: Min = {global_controls_min[i]:>8.4f}, Max = {global_controls_max[i]:>8.4f}"
        )

    print("=" * 80 + "\n")
    print(f"--- Plotting {len(valid_files)} valid trajectories ---")

    # ==========================================
    # FIGURE 1: STATES (8x3 Grid)
    # ==========================================
    fig_x, axs_x = plt.subplots(8, 3, figsize=(16, 20), sharex=True)
    fig_x.suptitle("Hexapod STATE Analysis (All Valid Runs)", fontsize=18, y=0.98)
    axs_x_flat = axs_x.flatten()

    for f in valid_files:
        states_data = states_dict[f]
        timesteps = np.arange(states_data.shape[0]) / 10.0  # Assuming 10Hz recording
        fname = os.path.basename(f)

        for i in range(24):
            axs_x_flat[i].plot(
                timesteps, states_data[:, i], label=fname, linewidth=1.5, alpha=0.8
            )

    for i in range(24):
        axs_x_flat[i].set_title(labels_x[i], fontsize=10)
        axs_x_flat[i].grid(True, alpha=0.3)
        if i < 3:
            axs_x_flat[i].set_ylabel("m", fontsize=8)
        else:
            axs_x_flat[i].set_ylabel("rad", fontsize=8)

    for i in range(21, 24):
        axs_x_flat[i].set_xlabel("Time (seconds)")

    # Global Legend Layout for States
    handles_x, leg_labels_x = axs_x_flat[0].get_legend_handles_labels()
    fig_x.subplots_adjust(right=0.85)
    fig_x.legend(
        handles_x,
        leg_labels_x,
        loc="center right",
        bbox_to_anchor=(0.99, 0.5),
        fontsize="small",
        title="File Names",
    )

    # ==========================================
    # FIGURE 2: CONTROLS (6x3 Grid)
    # ==========================================
    fig_u, axs_u = plt.subplots(6, 3, figsize=(16, 15), sharex=True)
    fig_u.suptitle("Hexapod CONTROL Analysis (All Valid Runs)", fontsize=18, y=0.98)
    axs_u_flat = axs_u.flatten()

    for f in valid_files:
        controls_data = controls_dict[f]
        # controls array might be 1 step shorter than states, so generate specific timesteps
        timesteps_u = np.arange(controls_data.shape[0]) / 10.0
        fname = os.path.basename(f)

        for i in range(18):
            axs_u_flat[i].plot(
                timesteps_u, controls_data[:, i], label=fname, linewidth=1.5, alpha=0.8
            )

    for i in range(18):
        axs_u_flat[i].set_title(labels_u[i], fontsize=10)
        axs_u_flat[i].grid(True, alpha=0.3)
        axs_u_flat[i].set_ylabel(
            "Cmd", fontsize=8
        )  # Change "Cmd" to "rad" or "Nm" based on your control type

    for i in range(15, 18):
        axs_u_flat[i].set_xlabel("Time (seconds)")

    # Global Legend Layout for Controls
    handles_u, leg_labels_u = axs_u_flat[0].get_legend_handles_labels()
    fig_u.subplots_adjust(right=0.85)
    fig_u.legend(
        handles_u,
        leg_labels_u,
        loc="center right",
        bbox_to_anchor=(0.99, 0.5),
        fontsize="small",
        title="File Names",
    )

    print("Plotting complete. Rendering windows...")
    plt.show()


if __name__ == "__main__":
    # Ensure this directory matches the output_dir in your collector
    analyze_hexapod_trajectories("hexapod_data/training")
