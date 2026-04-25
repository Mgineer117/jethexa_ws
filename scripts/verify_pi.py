"""
check_control_bounds.py
────────────────────────────────────────────────────────────────────────────
Loads a reference trajectory, adds noise to the state, queries the policy,
and checks if the resulting control u = u_ref + pi([x+noise, x_ref, u_ref])
violates the defined U_MIN / U_MAX bounds.

Usage
──────
  python check_control_bounds.py --path jethexa_data/training/my_traj.npz --noise 0.05 --algo ppo
"""

import argparse
import os
import sys
from pathlib import Path

# Resolve repo root and chdir there so "models/*.pth" and the
# `from models.policy_networks import ...` import keep working after the
# move into scripts/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

# Ensure this import path is correct for your project structure
from models.policy_networks import CLActor

matplotlib.use("Agg")

# ─── Bounds & Constants ──────────────────────────────────────────────────────
U_MIN = -5.0
U_MAX = 5.0
NUM_ACTUATORS = 18

# ─── Policy Wrapper ──────────────────────────────────────────────────────────


def load_policy(algo_name):
    if algo_name == "ppo":
        path = "models/ppo.pth"
        mode = "stochastic"
    elif algo_name == "carl":
        path = "models/carl.pth"
        mode = "stochastic"
    elif algo_name == "c3m":
        path = "models/c3m.pth"
        mode = "deterministic"
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}. Choose ppo, carl, or c3m.")

    actor = CLActor(
        x_dim=23,
        u_dim=18,
        num_windows=1,
        mode=mode,
    )

    # Check if file exists to prevent hard crash
    if not os.path.exists(path):
        sys.exit(f"[ERROR] Policy weights not found at: {path}")

    actor.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
    actor.eval()

    print(f"[INFO] Successfully loaded {algo_name.upper()} policy from {path}...")

    return actor


def get_policy_action(actor, x, x_ref, u_ref):
    """
    Constructs the observation and runs a forward pass through the PyTorch actor.
    """
    # 1. Construct the observation exactly as your environment does
    obs = np.concatenate([x, x_ref, u_ref])

    # 2. Convert to PyTorch tensor and add a batch dimension (shape: [1, obs_dim])
    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

    # 3. Get action from policy
    with torch.no_grad():
        out = actor(obs_tensor)

        # Handle different output structures depending on your CLActor implementation
        # (e.g., if it returns a tuple of (action, log_prob, etc.) or just the action)
        if isinstance(out, tuple):
            delta_u = out[0]
        else:
            delta_u = out

        # Remove batch dimension and convert back to numpy
        delta_u = delta_u.squeeze(0).numpy()

    return delta_u


# ─── Evaluation Logic ────────────────────────────────────────────────────────


def evaluate_trajectory(path: str, noise_std: float, actor: torch.nn.Module):
    if not os.path.isfile(path):
        sys.exit(f"[ERROR] File not found: {path}")

    data = np.load(path)

    # Handle p_z removal if your raw data still has 24 dims.
    # (Assuming it's pre-processed to 23 based on your x_dim=23 initialization)
    x_refs = data["states"]
    if x_refs.shape[1] == 24:
        x_refs = np.delete(x_refs, 2, axis=1)

    # Load reference controls
    if "controls" in data:
        u_refs = data["controls"]
    else:
        print("[WARNING] 'controls' not found in .npz! Defaulting to zeros.")
        u_refs = np.zeros((x_refs.shape[0], NUM_ACTUATORS))

    T = x_refs.shape[0]
    out_of_bounds_count = 0
    u_history = np.zeros((T, NUM_ACTUATORS))

    print(
        f"Evaluating {T} steps from {os.path.basename(path)} (noise std={noise_std})..."
    )

    for t in range(T):
        x_ref = x_refs[t]
        u_ref = u_refs[t]

        # 1. Simulate noisy state
        noise = np.random.normal(0, noise_std, size=x_ref.shape)
        x = x_ref + noise

        # 2. Query policy for delta_u
        delta_u = get_policy_action(actor, x, x_ref, u_ref)

        # 3. Calculate actual control sent to motors
        u = u_ref + delta_u
        u_history[t] = u

        # 4. Check for violations
        if np.any(u < U_MIN) or np.any(u > U_MAX):
            out_of_bounds_count += 1

    # ─── Report ───
    violation_rate = (out_of_bounds_count / T) * 100
    print("-" * 50)
    print("RESULTS:")
    print(f"Total Steps        : {T}")
    print(f"Violations         : {out_of_bounds_count} ({violation_rate:.2f}%)")
    print(f"Global Min Action  : {np.min(u_history):.3f} (Bound: {U_MIN})")
    print(f"Global Max Action  : {np.max(u_history):.3f} (Bound: {U_MAX})")
    print("-" * 50)

    return u_history


# ─── Plotting (Optional Visualization) ───────────────────────────────────────


def plot_control_bounds(u_history: np.ndarray, out_path="control_bounds.svg"):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0d1117")
    ax.set_facecolor("#161b22")

    T = u_history.shape[0]
    t_steps = np.arange(T)

    # Plot all 18 control signals
    for i in range(NUM_ACTUATORS):
        ax.plot(t_steps, u_history[:, i], alpha=0.4, linewidth=1.0)

    # Plot bounds
    ax.axhline(
        U_MAX, color="#f78166", linestyle="--", linewidth=2, label="U_MAX (+5.0)"
    )
    ax.axhline(
        U_MIN, color="#f78166", linestyle="--", linewidth=2, label="U_MIN (-5.0)"
    )

    # Shade out-of-bounds regions
    ax.axhspan(U_MAX, np.max(u_history) + 1, color="#f78166", alpha=0.1)
    ax.axhspan(np.min(u_history) - 1, U_MIN, color="#f78166", alpha=0.1)

    ax.set_title("Generated Controls $u = u_{ref} + \pi(obs)$", color="#e6edf3")
    ax.set_xlabel("Time Step", color="#8b949e")
    ax.set_ylabel("Control Signal (rad)", color="#8b949e")
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.legend(
        facecolor="#161b22",
        edgecolor="#30363d",
        labelcolor="#c9d1d9",
        loc="upper right",
    )

    plt.tight_layout()
    fig.savefig(out_path, format="svg", facecolor=fig.get_facecolor())
    print(f"Saved control visualization to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Check HexapodEnv control bounds with loaded policy."
    )
    parser.add_argument(
        "--path", type=str, required=True, help="Path to a single .npz trajectory file."
    )
    parser.add_argument(
        "--noise", type=float, default=0.01, help="Standard deviation of state noise."
    )
    parser.add_argument(
        "--algo",
        type=str,
        choices=["ppo", "carl", "c3m"],
        default="ppo",
        help="Which policy algorithm to load (ppo, carl, or c3m).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="control_bounds.svg",
        help="Output path for the plot.",
    )
    args = parser.parse_args()

    # Load the specified policy
    actor = load_policy(args.algo)

    # Evaluate the trajectory
    u_history = evaluate_trajectory(args.path, args.noise, actor)

    # Plot results
    plot_control_bounds(u_history, out_path=args.out)


if __name__ == "__main__":
    main()
