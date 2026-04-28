import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Define colors manually if __init__ is not available
from __init__ import N, COLORS, LABELS, LINESTYLES, SLOPES, AVG_ERROR, smooth

TIME_INTERVAL = 0.05

# --- Load Reference Trajectory ---
# Ensure test_data.json exists or handle the error
try:
    with open("test_data.json", "r") as f:
        data = json.load(f)
        xref_list = data["states"][0]
    xref_np = np.array(xref_list)
    max_len = xref_np.shape[0]
except FileNotFoundError:
    print(
        "Error: 'test_data.json' not found. Creating dummy reference for demonstration."
    )
    xref_np = np.zeros((100, 3))  # Dummy
    max_len = 100

# --- ALGORITHM ORDER ---
algo_names = ["cac", "c3m", "ppo"]
label_map = {"cac": "CARL", "c3m": "C3M", "ppo": "PPO"}

json_dict = {}
for algo in algo_names:
    if os.path.isdir(algo):
        json_files = [f for f in os.listdir(algo) if f.endswith(".json")]
        json_dict[algo] = json_files
    else:
        json_dict[algo] = []

stats = {
    "algo": [],
    "error_mean": [],
    "error_ci": [],
    "comp_mean": [],
    "comp_ci": [],
    "colors": [],
    "labels": [],
}

# --- Process Data ---
print("-" * 60)
for algo in algo_names:
    files = json_dict[algo]

    # If no files found, skip or use dummy data for plotting check
    if not files:
        print(f"[{algo}] No data found. Skipping.")
        continue

    error_metric_list = []
    completion_list = []

    for file in files:
        with open(os.path.join(algo, file), "r") as f:
            data = json.load(f)

        x_traj_list = data["x"]
        L_original = len(x_traj_list)
        if L_original == 0:
            continue

        # 1. Completion Percentage
        pct = min(100.0, (L_original / max_len) * 100.0)
        completion_list.append(pct)

        # 2. Standardize Length
        L_curr = L_original
        if L_curr >= max_len:
            x_traj_list = x_traj_list[:max_len]
            L_curr = max_len
        x_traj_np = np.array(x_traj_list)

        # 3. Normalized Tracking Error
        normalization_factor = np.linalg.norm(x_traj_np[0, :] - xref_np[0, :])
        if normalization_factor < 1e-6:
            normalization_factor = 1.0

        error = (x_traj_np - xref_np[:L_curr, :]) / normalization_factor
        error_norm = np.linalg.norm(error, axis=1)

        # 4. METRIC CALCULATION (m^2AUC)
        # Trapezoidal integration of error
        area = np.trapz(error_norm, dx=TIME_INTERVAL)

        # Apply the scaling factor (L_max / L)^2
        metric = area * ((max_len / L_curr) ** 2)

        error_metric_list.append(metric)

    if len(error_metric_list) > 0:
        stats["algo"].append(algo)
        stats["labels"].append(label_map.get(algo, algo))

        # Means and CIs (95%)
        e_mean = np.mean(error_metric_list)
        e_std = np.std(error_metric_list)
        e_ci = 1.96 * (e_std / np.sqrt(len(error_metric_list)))

        c_mean = np.mean(completion_list)
        c_std = np.std(completion_list)
        c_ci = 1.96 * (c_std / np.sqrt(len(completion_list)))

        stats["error_mean"].append(e_mean)
        stats["error_ci"].append(e_ci)
        stats["comp_mean"].append(c_mean)
        stats["comp_ci"].append(c_ci)
        stats["colors"].append(COLORS.get(algo, "gray"))

        print(
            f"[{algo.upper()}] m2AUC: {e_mean:.2f} ± {e_ci:.2f} | Completion: {c_mean:.1f}% ± {c_ci:.1f}%"
        )
print("-" * 60)

# --- Plotting (Scatter with Error Bars) ---
fig, ax = plt.subplots(figsize=(10, 7))

# Loop through each algorithm to plot them individually (for the legend)
for i, algo in enumerate(stats["algo"]):
    ax.errorbar(
        stats["comp_mean"][i],  # X: Completion
        stats["error_mean"][i],  # Y: m2AUC
        xerr=stats["comp_ci"][i],  # X Error
        yerr=stats["error_ci"][i],  # Y Error
        fmt="o",  # Marker shape (circle)
        markersize=15,  # Size of the dot
        capsize=10,  # Size of error bar caps
        capthick=4,
        elinewidth=5,
        color=stats["colors"][i],
        label=stats["labels"][i],  # Label for Legend
        alpha=0.9,
    )

# --- Formatting ---
ax.set_xlabel("Path-tracking Completion (%)", fontsize=32, labelpad=15)
ax.set_ylabel(r"m$^2$AUC", fontsize=32, labelpad=15)

# Ticks
ax.tick_params(axis="both", which="major", labelsize=18)
ax.grid(True, linestyle="--", alpha=0.6)

# Limits (Optional - Adjust based on your data)
# Ensure X axis goes up to 105 to show full completion clearly
ax.set_xlim(left=0, right=45)
ax.set_ylim(bottom=0, top=None)

# ax.set_xscale("linear")
# ax.set_yscale("log")

# --- Legend ---
# Placing legend in a clear spot (e.g., upper left or best fit)
ax.legend(
    fontsize=26,
    title="Algorithm",
    title_fontsize=32,
    loc="best",
    frameon=True,
    edgecolor="black",
    framealpha=0.9,
)

plt.tight_layout()

# Save
plt.savefig("scatter_metric_plot.svg")
plt.savefig("scatter_metric_plot.pdf")

print("Plot saved as 'scatter_metric_plot.svg' and 'scatter_metric_plot.pdf'")
