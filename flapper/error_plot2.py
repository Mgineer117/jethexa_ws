import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d  # No longer used, but kept for reference
from numpy.linalg import lstsq  # Import least squares

from __init__ import N, COLORS, LABELS, LINESTYLES, SLOPES, AVG_ERROR, smooth

TIME_INTERVAL = 0.05
EXTRAPOLATION_WINDOW = 10  # Use last 10 points to find the slope

# --- Load Reference Trajectory ---
with open("test_data.json", "r") as f:
    data = json.load(f)
    xref_list = data["states"][0]  # Keep as list for now
xref_np = np.array(xref_list)
max_len = xref_np.shape[0]  # This is the 'N' for normalization

algo_names = ["cac", "c3m", "ppo"]
json_dict = {}
for algo in algo_names:
    json_files = [f for f in os.listdir(algo) if f.endswith(".json")]
    json_dict[algo] = json_files

# --- Process Data ---
data_dict = {}
mauc_dict = {}  # Store MAUC strings (e.g., "0.5 ± 0.1") for the plot legend
lengths = {}  # Store minimum length of trajectories for each algo

for algo, files in json_dict.items():
    extrapolated_errors_list = []  # For plotting
    mauc_list = []  # For MAUC calculation
    length = []
    slopes = []

    for file in files:
        with open(os.path.join(algo, file), "r") as f:
            data = json.load(f)

        x_traj_list = data["x"]
        length.append(len(x_traj_list))
        L = len(x_traj_list)  # Original length of this trajectory

        if L == 0:
            continue  # Skip empty or corrupted files

        # Truncate long trajectories
        if L >= max_len:
            x_traj_list = x_traj_list[:max_len]
            L = max_len

        x_traj_np = np.array(x_traj_list)

        # --- Error Calculation ---
        normalization_factor = np.linalg.norm(x_traj_np[0, :] - xref_np[0, :])
        if normalization_factor < 1e-6:
            normalization_factor = 1.0  # Avoid division by zero

        error = (x_traj_np - xref_np[:L, :]) / normalization_factor
        error = np.linalg.norm(error, axis=1, keepdims=True)
        error = smooth(error, weight=0.9).squeeze(-1)  # 1D error array of length L

        # --- FIX: Least-Squares Non-Decreasing Extrapolation ---
        if L < max_len:
            # 1. Define window for slope calculation
            window = min(L, EXTRAPOLATION_WINDOW)

            if L < 2:
                # Not enough data for a slope, assume 0
                slope = 0.0
            else:
                # 2. Prepare data for least-squares (y = mx)
                y_data = error[-window:]

                # Create the 'x' data (time in seconds)
                x_data_time = np.arange(L - window, L) * TIME_INTERVAL

                # 3. Solve for slope m in y = m*x using least-squares
                # A must be a column vector (shape [window, 1]) for lstsq
                A = x_data_time[:, np.newaxis]
                try:
                    # m is a (1,) array containing the slope
                    m, _, _, _ = np.linalg.lstsq(A, y_data, rcond=None)
                    slope = m[0]
                except np.linalg.LinAlgError:
                    slope = 0.0  # Failsafe

            # 4. Enforce a non-decreasing (non-negative) trend
            slope = max(0.0, slope)
            slopes.append(slope)

            # 5. Create the new, full-length error array
            error_extrapolated = np.zeros(max_len)
            error_extrapolated[:L] = error  # Fill in the known data

            # 6. Manually extrapolate the new points
            last_time = (L - 1) * TIME_INTERVAL
            last_error_value = error[-1]

            for i in range(L, max_len):
                new_time = i * TIME_INTERVAL
                # Extrapolate: y_new = y_last + slope * (t_new - t_last)
                error_extrapolated[i] = last_error_value + slope * (
                    new_time - last_time
                )

        else:
            # This trajectory is already full-length
            error_extrapolated = error

        extrapolated_errors_list.append(error_extrapolated)

        # --- MAUC Calculation (using original, non-extrapolated data) ---
        auc = (max_len / L) * np.trapezoid(error_extrapolated, dx=TIME_INTERVAL)
        mauc_list.append(auc)

    lengths[algo] = max(length)  # Minimum length for this algo

    # --- Store results for this algo ---
    data_dict[algo] = np.array(extrapolated_errors_list)

    # --- Finalize MAUC score and save for plotting ---
    mauc_mean = np.mean(mauc_list)
    mauc_ci = 1.96 * (np.std(mauc_list) / np.sqrt(len(mauc_list)))
    mauc_string = r"{mauc_mean:.3f} $\pm$ {mauc_ci:.3f}"
    mauc_dict[algo] = mauc_string

    slope_mean = np.mean(slopes)
    slope_ci = 1.96 * (np.std(slopes) / np.sqrt(len(slopes)))
    print(f"{algo} slope / mAUC: {slope_mean:.6f} ± {slope_ci:.6f} / {mauc_string}")


# --- Plotting ---
plt.figure(figsize=(12, 8))

# Define a small positive number to be the "floor" of the log plot
epsilon = 1e-1

for algo, error in data_dict.items():
    time_steps = np.arange(error.shape[1]) * 0.05
    mean_error = np.nanmean(error, axis=0)
    std_error = np.nanstd(error, axis=0)
    valid_counts = np.sum(~np.isnan(error), axis=0)

    with np.errstate(invalid="ignore", divide="ignore"):
        ci95 = 1.96 * std_error / np.sqrt(valid_counts)
        ci95 = np.nan_to_num(ci95, nan=0.0)

    # Get the label (with MAUC score, assuming you've calculated it)
    # plot_label = f"{LABELS[algo]}"
    plot_label = f"{SLOPES[algo]}"
    # plot_label = f"{AVG_ERROR[algo]}"

    # Get the index where extrapolation starts (the shortest trajectory)
    split_index = lengths[algo]

    # --- Plot "Real" Data (Solid Line) ---
    plt.plot(
        time_steps[: split_index + 1],
        mean_error[: split_index + 1],
        color=COLORS[algo],
        linestyle=LINESTYLES[algo],
        linewidth=5,
        alpha=0.9,
    )

    # --- Plot "Extrapolated" Data (Dashed Line) --- #
    plt.plot(
        time_steps[split_index:],
        mean_error[split_index:],
        label=plot_label,  # Label goes on the main line
        color=COLORS[algo],
        linestyle="--",  # LINESTYLES[algo],
        linewidth=3,
        alpha=0.8,
    )

    # --- Plot CI for "Real" Data ---
    lower_bound_real = np.maximum(
        mean_error[: split_index + 1] - ci95[: split_index + 1], epsilon
    )
    upper_bound_real = mean_error[: split_index + 1] + ci95[: split_index + 1]

    plt.fill_between(
        time_steps[: split_index + 1],
        lower_bound_real,
        upper_bound_real,
        alpha=0.2,
        color=COLORS[algo],
    )

    # --- Plot CI for "Extrapolated" Data ---
    lower_bound_extra = np.maximum(
        mean_error[split_index:] - ci95[split_index:], epsilon
    )
    upper_bound_extra = mean_error[split_index:] + ci95[split_index:]

    plt.fill_between(
        time_steps[split_index:],
        lower_bound_extra,
        upper_bound_extra,
        alpha=0.1,  # Make extrapolated CI lighter
        color=COLORS[algo],
    )

    # ... (Your axvline and text code for extrapolation start) ...


# --- Finalize Plot --- #
# plt.title("Real-world experiments: Flapper Drone", fontsize=32)
plt.xlabel("Time (s)", fontsize=28)
plt.ylabel("Normalized Tracking Error", fontsize=28)
plt.ylim(0.5, None)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid(True, linestyle="--", alpha=0.7)

# Set the y-axis bottom and use log scale
plt.ylim(bottom=epsilon)
plt.yscale("log")

# plt.legend(title="mAUC", title_fontsize=28, fontsize=22, loc="lower right")
# plt.legend(
#     title=r"Error rate $\frac{d(x - x^*)}{dt}$",
#     title_fontsize=28,
#     fontsize=22,
#     loc="lower right",
# )

plt.legend(
    title=r"Linear Error Trend",
    title_fontsize=28,
    fontsize=22,
    loc="lower left",
)
# Save the figure *before* showing it
plt.savefig("mse_plot.svg")
plt.savefig("mse_plot.pdf")
