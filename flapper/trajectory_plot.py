import os
import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MaxNLocator  # Import for nbins

# NOTE: Removed 'COLORS' from the import
from __init__ import N, LABELS, LINESTYLES, smooth

# --- Load Reference Trajectory ---
with open("test_data.json", "r") as f:
    data = json.load(f)
    xref = data["states"][0]
xref = np.array(xref)
max_len = xref.shape[0]
STATE_DIM = xref.shape[1]

# --- Find Algorithm JSON Files ---
algo_names = ["cac", "ppo", "c3m"]
json_dict = {}
for algo in algo_names:
    json_files = [f for f in os.listdir(algo) if f.endswith(".json")]
    json_dict[algo] = json_files

# --- Process Data for All Algorithms ---
all_algo_trajectories = {}
mean_trajectories_dict = {}

for algo, files in json_dict.items():
    x_list_for_algo = []

    for file in files:
        with open(os.path.join(algo, file), "r") as f:
            data = json.load(f)
            x_traj = data["x"][: len(data["x"]) - 1]

        # --- Robustness/Padding/Truncating Logic (as before) ---
        current_len = len(x_traj)

        if current_len == 0:
            print(f"Warning: Empty trajectory in {os.path.join(algo, file)}. Skipping.")
            continue

        x_traj_np = np.array(x_traj)

        if x_traj_np.ndim != 2 or x_traj_np.shape[1] != STATE_DIM:
            print(
                f"Warning: Bad shape in {os.path.join(algo, file)}. Shape: {x_traj_np.shape}. Skipping."
            )
            continue

        if current_len < max_len:
            padded_traj_np = np.pad(
                x_traj_np, ((0, max_len - current_len), (0, 0)), mode="edge"
            )
        else:
            padded_traj_np = x_traj_np[:max_len, :]

        x_list_for_algo.append(padded_traj_np)

    if x_list_for_algo:
        all_algo_trajectories[algo] = np.array(x_list_for_algo)
        mean_trajectories_dict[algo] = np.mean(all_algo_trajectories[algo], axis=0)
        print(f"Processed {len(x_list_for_algo)} trajectories for {algo}.")
    else:
        print(f"No valid trajectories processed for {algo}.")


# ==========================================================
# --- Helper Function for New Plot Style ---
# ==========================================================
def apply_plot_style(ax, all_data_points):
    """Applies the font sizes, grid, ticks, and aspect ratio."""

    # --- THIS IS THE FIX ---
    # Labels with fontsize 28
    # Increased labelpad from 10 to 20 to prevent overlap
    ax.set_xlabel("X Position (m)", fontsize=16, labelpad=10)
    ax.set_ylabel("Y Position (m)", fontsize=16, labelpad=10)
    ax.set_zlabel("Z Position (m)", fontsize=16, labelpad=5)
    # --- END OF FIX ---

    # Ticks with fontsize 22
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=14)
    ax.tick_params(axis="z", labelsize=14)

    # Set number of ticks (nbins=5)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.zaxis.set_major_locator(MaxNLocator(nbins=5))

    # Grid style from snippet
    ax.grid(True, linestyle="--", alpha=0.3)

    # Set the viewing angle
    ax.view_init(elev=10, azim=15)

    # Set fixed X and Y limits as requested
    ax.set_xlim([-1.0, 0.5])
    ax.set_ylim([-1.2, 1.0])
    ax.set_zlim([0.0, 1.65])

    # Auto-scale Z axis based on all data
    # z_min, z_max = all_data_points[:, 2].min(), all_data_points[:, 2].max()
    # z_padding = (z_max - z_min) * 0.1  # Add 10% padding
    # if z_padding == 0:  # Handle flat data
    # z_padding = 0.1
    # ax.set_zlim([z_min - z_padding, z_max + z_padding])


# Concatenate all data for global plot limits (still needed for Z-axis)
all_points_for_limits = [xref[:, :3]]
for algo_runs in all_algo_trajectories.values():
    all_points_for_limits.append(algo_runs[:, :, :3].reshape(-1, 3))
global_all_points = np.concatenate(all_points_for_limits)


# ==========================================================
# --- Plotting Individual Trajectories for Each Algorithm ---
# ==========================================================
for algo, trajectories_3d_array in all_algo_trajectories.items():
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the reference trajectory
    ax.plot(
        xref[:, 0],
        xref[:, 1],
        xref[:, 2],
        label="Reference",
        color="black",
        linestyle="--",
        linewidth=2,
        alpha=0.8,
    )
    ax.scatter(
        xref[0, 0], xref[0, 1], xref[0, 2], color="black", marker="*", s=100, alpha=0.8
    )
    # ax.scatter(xref[-1, 0], xref[-1, 1], xref[-1, 2], color="red", marker="o", s=75)

    # Plot each individual trajectory
    for i in range(5):
        label = "Trajectories" if i == 0 else None
        traj = trajectories_3d_array[i]

        ax.plot(
            traj[:, 0],
            traj[:, 1],
            traj[:, 2],
            label=label,
            alpha=0.9,
            linewidth=3,
        )

        ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], marker="*", s=120, alpha=0.9)

    # --- Set Title ---
    # if algo == "cac":
    #     ax.set_title(f"Trajectories for {algo.upper()} (ours)", fontsize=32)
    # else:
    #     ax.set_title(f"Trajectories for {algo.upper()}", fontsize=32)

    apply_plot_style(ax, global_all_points)
    # ax.legend(fontsize=22, loc='upper left')

    # --- Apply tight_layout ---
    fig.tight_layout()
    plt.savefig(f"{algo}_individual_trajectories_3d.svg")
    plt.savefig(f"{algo}_individual_trajectories_3d.pdf")
    print(f"Saved individual trajectory plot for {algo}.")

# ==========================================================
# --- Single Combined Plot for Mean Trajectories ---
# ==========================================================
fig_combined = plt.figure(figsize=(12, 10))
ax_combined = fig_combined.add_subplot(111, projection="3d")

# Plot the reference trajectory
ax_combined.plot(
    xref[:, 0],
    xref[:, 1],
    xref[:, 2],
    label="Reference",
    color="black",
    linestyle="--",
    linewidth=3,
    alpha=0.8,
)
ax_combined.scatter(xref[0, 0], xref[0, 1], xref[0, 2], color="black", marker="o", s=60)
ax_combined.scatter(
    xref[-1, 0], xref[-1, 1], xref[-1, 2], color="black", marker="x", s=60
)


# --- COLORMAP LOGIC ---
num_algos = len(mean_trajectories_dict)
algo_colormap = plt.colormaps["tab20b"]
algo_colors = algo_colormap(np.linspace(0, 1, num_algos))
algo_color_map = {
    algo: algo_colors[i] for i, algo in enumerate(mean_trajectories_dict.keys())
}
# ---

# Plot the mean trajectory for each algorithm
for algo, mean_traj in mean_trajectories_dict.items():
    ax_combined.plot(
        mean_traj[:, 0],
        mean_traj[:, 1],
        mean_traj[:, 2],
        label=LABELS[algo],
        color=algo_color_map[algo],
        linestyle=LINESTYLES[algo],
        linewidth=4,
    )

    ax_combined.scatter(
        mean_traj[0, 0],
        mean_traj[0, 1],
        mean_traj[0, 2],
        color="black",
        marker="o",
        s=60,
        alpha=1,
    )
    ax_combined.scatter(
        mean_traj[-1, 0],
        mean_traj[-1, 1],
        mean_traj[-1, 2],
        color="black",
        marker="x",
        s=60,
        alpha=1,
    )

# Apply style to combined plot
ax_combined.set_title("Mean Trajectories Comparison", fontsize=32)
apply_plot_style(ax_combined, global_all_points)
ax_combined.legend(fontsize=22)

# --- Apply tight_layout ---
fig_combined.tight_layout()
plt.savefig("combined_mean_trajectories_3d.svg")
plt.savefig("combined_mean_trajectories_3d.pdf")
print("Saved combined mean trajectory plot.")
