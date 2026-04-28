import os
import json
import numpy as np
import matplotlib.pyplot as plt

from __init__ import N, COLORS, LABELS, LINESTYLES, smooth

with open("test_data.json", "r") as f:
    data = json.load(f)
    xref = data["states"][0]
xref = np.array(xref)
max_len = xref.shape[0]


algo_names = ["cac", "ppo", "c3m"]
# access to all json files in each directory name of algo
json_dict = {}
for algo in algo_names:
    json_files = [f for f in os.listdir(algo) if f.endswith(".json")]
    json_dict[algo] = json_files


# construct current target matrix
data_dict = {}
lengths = {}
for algo, files in json_dict.items():
    x_list = []  # Use a temporary list

    # First loop: load all data and find the true max length
    for file in files:
        with open(os.path.join(algo, file), "r") as f:
            data = json.load(f)

            x_traj = data["x"]
            lengths[algo] = len(x_traj)
            if len(x_traj) < max_len:
                for k in range(len(x_traj), max_len):
                    x_traj.append(x_traj[lengths[algo] - 1])  # Pad with last state
            else:
                x_traj = x_traj[:max_len]  # Truncate to max_len

            x_list.append(x_traj)

    # find errors
    x = np.array(x_list)
    normalization_factor = np.linalg.norm(x[:, 0] - xref[0], axis=1, keepdims=True)[
        ..., np.newaxis
    ]
    error = (x - xref[np.newaxis, :, :]) / normalization_factor
    error = np.linalg.norm(error, axis=2)
    error = smooth(error, weight=0.9)
    data_dict[algo] = error

    # compute mauc
    ##### USE EXTRAPOLATION #####
    mauc_list = []
    for i in range(error.shape[0]):
        auc = (N / lengths[algo]) * np.trapezoid(
            error[i][: lengths[algo]], dx=0.05
        )  # Assuming time step of 0.05s
        mauc_list.append(auc)
    print(
        f"{algo} mAUC: {np.mean(mauc_list):.3f} ± {1.96*(np.std(mauc_list) / np.sqrt(len(mauc_list))):.3f}"
    )

print(lengths)
# Plot mean and 95 % confidence interval
i = 0
plt.figure(figsize=(12, 8))
for algo, error in data_dict.items():
    time_steps = np.arange(error.shape[1]) * 0.05
    mean_error = np.mean(error, axis=0)
    std_error = np.std(error, axis=0)
    ci95 = 1.96 * std_error / np.sqrt(error.shape[0])
    plt.plot(
        time_steps[: lengths[algo]],
        mean_error[: lengths[algo]],
        label=LABELS[algo],
        color=COLORS[algo],
        linestyle=LINESTYLES[algo],
        linewidth=5,
    )
    plt.fill_between(
        time_steps[: lengths[algo]],
        mean_error[: lengths[algo]] - ci95[: lengths[algo]],
        mean_error[: lengths[algo]] + ci95[: lengths[algo]],
        alpha=0.2,
        color=COLORS[algo],
    )
    i += 1
plt.title("Real-world experiments: Flapper Drone", fontsize=32)
plt.xlabel("Time (s)", fontsize=28)
plt.ylabel("Normalized Tracking Error", fontsize=28)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(title="mAUC", title_fontsize=28, fontsize=22)
plt.savefig("mse_plot.svg")
plt.savefig("mse_plot.pdf")
