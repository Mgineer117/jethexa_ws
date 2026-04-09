#!/usr/bin/env python3
"""
JET-HEXA TRAJECTORY ANALYSIS (INTEGRATION VS ACTUAL)
----------------------------
Functionality:
1. Loads the most recent .npz rollout/trajectory data.
2. Extracts actual recorded joint positions.
3. Integrates the 'controls' array to generate the virtual commanded trajectory.
4. Plots Actual vs Integrated side-by-side for all 18 joints.
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def load_latest_data(directory="hexapod_data", filename=None):
    if filename:
        filepath = os.path.join(directory, filename)
        if not os.path.exists(filepath):
            print(f"[ERROR]: File not found: {filepath}")
            return None
        return np.load(filepath)

    files = glob.glob(os.path.join(directory, "*.npz"))
    if not files:
        print(f"[ERROR]: No .npz files found in: {directory}")
        return None
    latest_file = max(files, key=os.path.getmtime)
    print(f"[INFO]: Loading data from {latest_file}")
    return np.load(latest_file)


def load_test_data():
    """
    24-dim State Vector (Updated):
    - [0:3]   : Position (x,y,z)
    - [3:6]   : Orientation (\theta, \phi, \psi)
    - [6:24]  : Joint Positions (18 DOF)
    """

    test_file = os.path.join("models/test_traj.npz")
    if not os.path.exists(test_file):
        return None
    return np.load(test_file)


def plot_integration_comparison(data, hz=10.0):
    dt = 1.0 / hz

    # Extract data
    # Note: Adjust these indices if your state vector format changes
    actual_joints = data["states"][:, 6:24]
    controls = data["controls"]
    total_steps = len(actual_joints)

    print(f"[INFO]: Processing {total_steps} timesteps...")

    # --- INTEGRATION LOGIC ---
    integrated_joints = np.zeros_like(actual_joints)

    # q0 = q0 (Start the integration exactly at the first recorded physical state)
    integrated_joints[0] = actual_joints[0]

    # q_t = q_{t-1} + control * dt
    for i in range(1, total_steps):
        # We use controls[i] assuming the control at step i moves us to state i
        # (If your controls[i-1] moves you to state i, change this to controls[i-1])
        integrated_joints[i] = integrated_joints[i - 1] + (controls[i] * dt)

    # --- PLOTTING LOGIC (6x3 GRID) ---
    print("[INFO]: Generating plots...")
    fig, axs = plt.subplots(6, 3, figsize=(16, 12), sharex=True)

    leg_names = [
        "Leg 1 (RF)",
        "Leg 2 (LF)",
        "Leg 3 (RM)",
        "Leg 4 (LM)",
        "Leg 5 (RH)",
        "Leg 6 (LH)",
    ]
    joint_names = ["Coxa (J0)", "Femur (J1)", "Tibia (J2)"]

    # Time array for the X-axis
    time_steps = np.arange(total_steps)

    for leg_idx in range(6):
        for joint_idx in range(3):
            ax = axs[leg_idx, joint_idx]
            global_joint_idx = (leg_idx * 3) + joint_idx

            # Plot Actual vs Integrated
            ax.plot(
                time_steps,
                actual_joints[:, global_joint_idx],
                color="tab:blue",
                label="Actual State (q)",
                linewidth=2,
            )

            ax.plot(
                time_steps,
                integrated_joints[:, global_joint_idx],
                color="tab:orange",
                linestyle="--",
                label="Integrated Cmd",
                linewidth=2,
            )

            # Formatting: Titles
            if leg_idx == 0:
                ax.set_title(joint_names[joint_idx], fontweight="bold")

            # Formatting: Row Labels
            if joint_idx == 0:
                ax.set_ylabel(
                    leg_names[leg_idx],
                    fontweight="bold",
                    rotation=0,
                    labelpad=45,
                    ha="right",
                )

            ax.grid(True, alpha=0.3)

            # Add legend only to the top right plot to avoid clutter
            if leg_idx == 0 and joint_idx == 2:
                ax.legend(loc="upper right", bbox_to_anchor=(1.05, 1.25))

    # Add X-axis labels to the bottom row
    for j in range(3):
        axs[5, j].set_xlabel("Time Step")

    plt.suptitle(
        "Trajectory Verification: Actual States vs Integrated Virtual Commands",
        fontsize=16,
        y=0.98,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, left=0.15, right=0.95)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze hexapod trajectory integration."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="hexapod_data",
        help="Directory containing .npz files",
    )
    parser.add_argument(
        "--file", type=str, default=None, help="Specific file to load (optional)"
    )
    parser.add_argument(
        "--hz", type=float, default=10.0, help="Control loop frequency (Hz)"
    )
    args = parser.parse_args()

    # directory=args.dir, filename=args.file
    data_archive = load_test_data()

    if data_archive is not None:
        plot_integration_comparison(data_archive, hz=args.hz)
