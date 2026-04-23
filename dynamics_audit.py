#!/usr/bin/env python3
"""
dynamics_audit.py
─────────────────────────────────────────────────────────────────────────────
Safety-first audit of the learned Jacobian dynamics vs. the real robot.

Runs the same control loop as control_loop.py but maintains a SECOND,
purely-simulated trajectory that feeds the model's own predicted state back
to the policy (closed-loop approx rollout), and PAUSES after every step so
the operator can inspect and abort before danger.

Two parallel rollouts
─────────────────────
  REAL   – x comes from robot sensors; u goes to the robot hardware.
  APPROX – x comes from the Jacobian dynamics model (x_approx feeds the
            policy at every step, errors compound over the horizon).

Three output figures
────────────────────
  Fig 1  Per-timestep one-step prediction error, per state dimension.
         Shows how well the model predicts ONE step from the real state.
  Fig 2  Cumulative L2 divergence ‖x_real[t] – x_approx[t]‖ over time.
         Shows error accumulation when you trust the model entirely.
  Fig 3  2-D XY trajectory overlay (real, approx, reference).

Usage
─────
  python dynamics_audit.py --algo ppo --control-scaler 0.1 --duration 30
  python dynamics_audit.py --algo carl --traj-path hexapod_data/training/foo.npz
"""

import argparse
import os
import sys
import time
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # swap to "TkAgg" if you have a live display
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import rospy

from base import Base
from jethexa_controller_interfaces.msg import JointCommand
from models.policy_networks import CLActor

# ─── State/control dimensions ─────────────────────────────────────────────────
X_DIM = 23
U_DIM = 18
ATTITUDE_IDX = [2, 3, 4]   # phi, theta, psi – wrap-aware difference

STATE_LABELS = (
    ["px", "py", "phi", "theta", "psi"]
    + [f"q{i+1}" for i in range(18)]
)


# ─── Jacobian dynamics model ──────────────────────────────────────────────────
class HexapodDynamics(nn.Module):
    """
    Learned base-pose Jacobian J(x) s.t.  x_dot_base ≈ J(x) @ u.
    Full dynamics: x_next = x + dt * [J(x); I_18] @ u
    Lives permanently on CPU (tiny MLP, not worth GPU overhead).
    """

    def __init__(self, weights_path: str = "models/Jethexa_jacobian_net.pth"):
        super().__init__()
        nn_input_dim = X_DIM + len(ATTITUDE_IDX)  # angle-encoded input

        self.jacobian_net = nn.Sequential(
            nn.Linear(nn_input_dim, 64), nn.Tanh(),
            nn.Linear(64, 64),           nn.Tanh(),
            nn.Linear(64, 5 * U_DIM),
        )
        self.jacobian_net.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )
        self.to(torch.device("cpu"))

    def _encode_angles(self, x: torch.Tensor) -> torch.Tensor:
        """Replace each attitude angle θ with [cos θ, sin θ]."""
        parts, cur = [], 0
        for idx in ATTITUDE_IDX:
            if cur < idx:
                parts.append(x[:, cur:idx])
            theta = x[:, idx: idx + 1]
            parts += [torch.cos(theta), torch.sin(theta)]
            cur = idx + 1
        if cur < x.shape[1]:
            parts.append(x[:, cur:])
        return torch.cat(parts, dim=1)

    def forward(self, x) -> torch.Tensor:
        """Return Jacobian J(x) with shape (batch, 5, 18)."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        if x.dim() == 1:
            x = x.unsqueeze(0)
        enc = self._encode_angles(x.to("cpu"))
        return self.jacobian_net(enc).view(-1, 5, U_DIM)


# ─── Dynamics helpers ─────────────────────────────────────────────────────────
def model_step(x: np.ndarray, u: np.ndarray, Jb: np.ndarray, dt: float) -> np.ndarray:
    """
    Euler integration of the control-affine system:
        x_next = x + dt * B(x) @ u
    where B = [Jb (5×18); I_18] giving the 23-dim next state.
    """
    B = np.vstack([Jb, np.eye(U_DIM, dtype=np.float32)])  # (23, 18)
    return x + dt * (B @ u)


def angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise a – b, with wrap correction for attitude indices."""
    d = a - b
    for idx in ATTITUDE_IDX:
        d[..., idx] = np.arctan2(np.sin(d[..., idx]), np.cos(d[..., idx]))
    return d


# ─── ROS config (mirrors control_loop.py) ────────────────────────────────────
_ROS_CFG = {
    "use_mocap": True,
    "use_imu": False,
    "listen_joint_states": True,
    "use_abs_joint_commands": True,
    "use_rel_joint_commands": False,
    "load_test_traj": True,
    "load_recent_traj": False,
}


# ─── Main auditor class ───────────────────────────────────────────────────────
class DynamicsAuditor(Base):
    def __init__(self, algo_name: str, traj_path: Optional[str] = None):
        rospy.init_node("dynamics_auditor", anonymous=True)
        super().__init__(config=_ROS_CFG)

        self.policy = self._load_policy(algo_name)
        self.dyn = HexapodDynamics()
        self.dyn.eval()
        self._traj_path = traj_path

    # ── policy ────────────────────────────────────────────────────────────────
    def _load_policy(self, algo_name: str) -> CLActor:
        table = {
            "ppo":  ("models/ppo.pth",  "stochastic"),
            "carl": ("models/carl.pth", "deterministic"),
            "c3m":  ("models/c3m.pth",  "deterministic"),
        }
        if algo_name not in table:
            raise ValueError(f"Unknown algorithm '{algo_name}'. Choose: ppo, carl, c3m.")
        path, mode = table[algo_name]
        if not os.path.exists(path):
            sys.exit(f"[ERROR] Policy weights not found: {path}")

        actor = CLActor(x_dim=X_DIM, u_dim=U_DIM, num_windows=1, mode=mode)
        actor.load_state_dict(torch.load(path, map_location="cpu"))
        actor.eval()
        rospy.loginfo(f"[AUDIT] Loaded {algo_name.upper()} policy from {path}")
        return actor

    # ── ref trajectory ────────────────────────────────────────────────────────
    def _load_ref_traj(self):
        if self._traj_path:
            if not os.path.exists(self._traj_path):
                sys.exit(f"[ERROR] Trajectory file not found: {self._traj_path}")
            data = np.load(self._traj_path)
        else:
            data = self.test_data

        xref = data["states"]
        if xref.shape[1] == 24:          # strip p_z if present
            xref = np.delete(xref, 2, axis=1)
        xref = xref[-300:]
        uref = data["controls"][-300:]
        assert xref.shape == (300, X_DIM), f"Unexpected xref shape {xref.shape}"
        return xref, uref

    # ── policy query ──────────────────────────────────────────────────────────
    def _policy_u(
        self,
        x: np.ndarray,
        xref_t: np.ndarray,
        uref_t: np.ndarray,
        scaler: float,
    ) -> np.ndarray:
        obs = torch.from_numpy(
            np.concatenate([x, xref_t, uref_t]).astype(np.float32)
        ).unsqueeze(0)
        with torch.no_grad():
            du, _ = self.policy(obs)
            if torch.isnan(du).any():
                du = torch.zeros_like(du)
        du = scaler * du.cpu().numpy().squeeze()
        return np.clip(uref_t + du, -4.0, 4.0)

    # ── Jacobian lookup ───────────────────────────────────────────────────────
    def _jacobian(self, x: np.ndarray) -> np.ndarray:
        return self.dyn(x).detach().cpu().numpy().squeeze(0)  # (5, 18)

    # ── per-step console output ───────────────────────────────────────────────
    @staticmethod
    def _print_step(
        step: int,
        one_step_err: np.ndarray,
        traj_div: float,
    ):
        l2 = np.linalg.norm(one_step_err)
        top3_idx = np.argsort(np.abs(one_step_err))[::-1][:3]
        top3_str = "  ".join(
            f"{STATE_LABELS[j]}={one_step_err[j]:+.4f}" for j in top3_idx
        )
        sep = "─" * 65
        print(f"\n{sep}")
        print(
            f"  Step {step:3d} │ 1-step L2 error = {l2:.5f} │ "
            f"traj divergence = {traj_div:.5f}"
        )
        print(f"  Top-3 one-step errors:  {top3_str}")
        print(f"  Base (0-4):  {one_step_err[:5]}")
        print(f"  Joints (5-22) max abs: {np.max(np.abs(one_step_err[5:])):.4f}")
        print(f"{sep}")

    # ── main loop ─────────────────────────────────────────────────────────────
    def run_audit(self, control_scaler: float, duration: float = 30.0):
        xref, uref = self._load_ref_traj()
        total_steps = min(len(xref), int(duration / self.dt))

        # log buffers
        real_log      = []   # real states  (N, 23)
        approx_log    = []   # approx states (N, 23)
        one_step_errs = []   # x_real_next – x_model_pred  (N, 23)

        # both trajectories start at the current real state
        x_real   = np.concatenate([self.base_pos, self.base_attitude, self.joint_pos])
        x_approx = x_real.copy()

        rospy.loginfo(
            f"[AUDIT] Starting {total_steps}-step audit.  "
            "After each step press Enter to continue or 'q'<Enter> to abort."
        )

        for i in range(total_steps):
            if rospy.is_shutdown():
                break

            # 1. Read the real robot state
            x_real = np.concatenate([self.base_pos, self.base_attitude, self.joint_pos])
            real_log.append(x_real.copy())
            approx_log.append(x_approx.copy())

            # 2. Policy on real state → compute command → send to hardware
            u_real   = self._policy_u(x_real, xref[i], uref[i], control_scaler)
            Jb_real  = self._jacobian(x_real)
            target   = self.check_valid_joint_angle(
                self.joint_pos + u_real * self.dt, terminate_on_invalid=False
            )
            msg          = JointCommand()
            msg.target   = target.tolist()
            msg.duration = self.duration
            self.joint_abs_pub.publish(msg)

            # 3. Model prediction from the current REAL state (single-step test)
            x_model_pred = model_step(x_real, u_real, Jb_real, self.dt)

            # 4. Approx closed-loop: policy queries x_approx, model propagates it
            u_approx = self._policy_u(x_approx, xref[i], uref[i], control_scaler)
            Jb_approx = self._jacobian(x_approx)
            x_approx  = model_step(x_approx, u_approx, Jb_approx, self.dt)

            # 5. Wait one control cycle so the robot reaches the target
            self.rate.sleep()

            # 6. Read real next state → one-step mismatch
            x_real_next = np.concatenate([self.base_pos, self.base_attitude, self.joint_pos])
            err = angle_diff(x_real_next, x_model_pred)   # (23,)
            one_step_errs.append(err)

            # 7. Divergence between the two parallel trajectories
            traj_div = np.linalg.norm(angle_diff(x_real_next, x_approx))

            # 8. Per-step console summary
            self._print_step(i, err, traj_div)

            # 9. Safety pause — operator decides whether to continue
            try:
                choice = input(
                    "  >> [Enter] continue  |  [q] quit  |  [s] save & quit  > "
                ).strip().lower()
            except EOFError:
                choice = ""

            if choice in ("q", "s"):
                rospy.logwarn("[AUDIT] Operator aborted.")
                break

        # ── Post-run ──────────────────────────────────────────────────────────
        real_log      = np.array(real_log)
        approx_log    = np.array(approx_log)
        one_step_errs = np.array(one_step_errs)

        self._save_log(real_log, approx_log, one_step_errs)
        self._qualitative_report(one_step_errs, real_log, approx_log)
        self._plot_all(real_log, approx_log, one_step_errs, xref)

    # ── save log ──────────────────────────────────────────────────────────────
    @staticmethod
    def _save_log(
        real_log: np.ndarray,
        approx_log: np.ndarray,
        one_step_errs: np.ndarray,
    ):
        os.makedirs("hexapod_data", exist_ok=True)
        ts   = int(time.time())
        path = f"hexapod_data/dynamics_audit_{ts}.npz"
        np.savez_compressed(
            path,
            x_real=real_log,
            x_approx=approx_log,
            one_step_err=one_step_errs,
        )
        print(f"\n[AUDIT] Log saved → {path}")

    # ── qualitative report ────────────────────────────────────────────────────
    @staticmethod
    def _qualitative_report(
        one_step_errs: np.ndarray,
        real_log: np.ndarray,
        approx_log: np.ndarray,
    ):
        T    = len(one_step_errs)
        mae  = np.mean(np.abs(one_step_errs), axis=0)
        std  = np.std(one_step_errs,           axis=0)
        maxe = np.max(np.abs(one_step_errs),   axis=0)

        div_series = np.array([
            np.linalg.norm(angle_diff(real_log[t], approx_log[t]))
            for t in range(min(T, len(approx_log)))
        ])

        print("\n" + "═" * 70)
        print("  QUALITATIVE DYNAMICS MISMATCH REPORT")
        print("═" * 70)
        print(f"  {'Dim':<5}  {'Label':<7}  {'MAE':>9}  {'Std':>9}  {'MaxAbs':>9}")
        print("  " + "─" * 44)

        for j in np.argsort(mae)[::-1]:
            marker = " ◄" if mae[j] == mae.max() else ""
            print(
                f"  [{j:2d}]   {STATE_LABELS[j]:<7}  "
                f"{mae[j]:9.5f}  {std[j]:9.5f}  {maxe[j]:9.5f}{marker}"
            )

        base_mae  = mae[:5].mean()
        joint_mae = mae[5:].mean()

        print()
        print(f"  Base-pose (dims 0–4)  average MAE : {base_mae:.5f}")
        print(f"  Joints    (dims 5–22) average MAE : {joint_mae:.5f}")

        if len(div_series) > 1:
            peak_t  = int(np.argmax(div_series))
            accel   = np.diff(div_series)
            accel_t = int(np.argmax(accel))
            print(f"\n  Approx-traj divergence peak   : step {peak_t} "
                  f"(L2 = {div_series[peak_t]:.5f})")
            print(f"  Fastest divergence growth     : step {accel_t}→{accel_t+1} "
                  f"(Δ = {accel[accel_t]:.5f})")

        print()
        # Actionable interpretation
        ratio = base_mae / (joint_mae + 1e-9)
        if ratio > 5:
            print("  [!] Base-pose error dominates.  The Jacobian network under-predicts")
            print("      base translation/rotation.  Consider: more training data with")
            print("      aggressive manoeuvres, or adding base-velocity to the state.")
        elif ratio < 0.2:
            print("  [!] Joint-space error dominates.  The identity-block assumption")
            print("      (q_dot = u) may not hold at this dt.  Reduce dt or add joint")
            print("      velocity observations.")
        else:
            print("  [OK] Base and joint errors are in the same order of magnitude.")

        # Bias check
        bias = np.mean(one_step_errs, axis=0)
        biased = np.where(np.abs(bias) > 0.3 * mae)[0]
        if len(biased):
            labels = [STATE_LABELS[j] for j in biased]
            print(f"\n  [!] Systematic bias detected in: {labels}")
            print("      The model consistently over- or under-predicts these dims.")

        print("═" * 70)

    # ── figures ───────────────────────────────────────────────────────────────
    @staticmethod
    def _plot_all(
        real_log: np.ndarray,
        approx_log: np.ndarray,
        one_step_errs: np.ndarray,
        xref: np.ndarray,
    ):
        ts = int(time.time())
        os.makedirs("hexapod_data", exist_ok=True)
        T  = len(one_step_errs)
        T2 = len(real_log)
        t_ax = np.arange(T)

        # ── Fig 1: per-dim one-step error grid ────────────────────────────────
        ncols = 5
        nrows = 5   # 5×5 = 25 ≥ 23
        fig1, axes = plt.subplots(nrows, ncols, figsize=(20, 16), sharex=True)
        axes = axes.flatten()

        for j in range(X_DIM):
            ax = axes[j]
            ax.plot(t_ax, one_step_errs[:, j], color="#58a6ff", linewidth=0.9)
            ax.axhline(0, color="#8b949e", linewidth=0.6, linestyle="--")
            ax.set_title(STATE_LABELS[j], fontsize=9, pad=2)
            ax.tick_params(labelsize=6)
            # shade ±1 std band
            mu  = one_step_errs[:, j].mean()
            sig = one_step_errs[:, j].std()
            ax.axhspan(mu - sig, mu + sig, color="#58a6ff", alpha=0.08)
        for j in range(X_DIM, len(axes)):
            axes[j].set_visible(False)

        fig1.suptitle(
            "One-step Prediction Error  (x_real_next – x_model_pred)\n"
            "Grey band = ±1 std over the run",
            fontsize=12,
        )
        fig1.tight_layout(rect=[0, 0, 1, 0.95])
        out1 = f"hexapod_data/audit_onestep_{ts}.png"
        fig1.savefig(out1, dpi=150)
        plt.close(fig1)
        print(f"[AUDIT] Fig 1 → {out1}")

        # ── Fig 2: cumulative divergence ──────────────────────────────────────
        div = np.array([
            np.linalg.norm(angle_diff(real_log[t], approx_log[t]))
            for t in range(T2)
        ])
        # component-wise breakdown: base vs joints
        div_base  = np.array([
            np.linalg.norm(angle_diff(real_log[t, :5], approx_log[t, :5]))
            for t in range(T2)
        ])
        div_joint = np.array([
            np.linalg.norm(real_log[t, 5:] - approx_log[t, 5:])
            for t in range(T2)
        ])

        fig2, (ax2a, ax2b) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        ax2a.plot(np.arange(T2), div, color="#3fb950", linewidth=1.5,
                  label="Total ‖x_real – x_approx‖₂")
        ax2a.fill_between(np.arange(T2), 0, div, alpha=0.15, color="#3fb950")
        ax2a.set_ylabel("L2 divergence")
        ax2a.set_title("Cumulative Trajectory Divergence: Real vs Approx (closed-loop)")
        ax2a.legend()

        ax2b.plot(np.arange(T2), div_base,  label="Base pose (dims 0–4)",  color="#f78166")
        ax2b.plot(np.arange(T2), div_joint, label="Joints (dims 5–22)", color="#d2a8ff")
        ax2b.set_xlabel("Time step")
        ax2b.set_ylabel("Component L2")
        ax2b.set_title("Divergence Breakdown: Base vs Joints")
        ax2b.legend()

        fig2.tight_layout()
        out2 = f"hexapod_data/audit_divergence_{ts}.png"
        fig2.savefig(out2, dpi=150)
        plt.close(fig2)
        print(f"[AUDIT] Fig 2 → {out2}")

        # ── Fig 3: 2-D XY trajectory overlay ─────────────────────────────────
        fig3, ax3 = plt.subplots(figsize=(9, 9))
        N = min(T2, len(xref))
        if xref is not None and N > 0:
            ax3.plot(xref[:N, 0], xref[:N, 1],
                     color="#8b949e", linewidth=1.2, linestyle="--",
                     label="Reference")
        ax3.plot(real_log[:, 0],   real_log[:, 1],
                 color="#58a6ff", linewidth=1.8, label="Real robot")
        ax3.plot(approx_log[:, 0], approx_log[:, 1],
                 color="#f78166", linewidth=1.8, linestyle=":", label="Approx (model)")

        # mark start and end
        ax3.scatter(*real_log[0, :2],    color="#58a6ff", s=100, zorder=6,
                    edgecolors="white", linewidths=0.8, label="Start (real)")
        ax3.scatter(*real_log[-1, :2],   color="#58a6ff", marker="D", s=100, zorder=6,
                    edgecolors="white", linewidths=0.8, label="End (real)")
        ax3.scatter(*approx_log[-1, :2], color="#f78166", marker="x", s=120, zorder=6,
                    linewidths=2, label="End (approx)")

        ax3.set_xlabel("X (m)")
        ax3.set_ylabel("Y (m)")
        ax3.set_title("2-D XY Trajectory: Real vs Approx (model closed-loop)")
        ax3.legend(loc="best")
        ax3.set_aspect("equal", adjustable="datalim")
        fig3.tight_layout()
        out3 = f"hexapod_data/audit_trajectory_{ts}.png"
        fig3.savefig(out3, dpi=150)
        plt.close(fig3)
        print(f"[AUDIT] Fig 3 → {out3}")


# ─── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Per-step dynamics audit: real robot vs Jacobian model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--algo-name",      type=str,   default="ppo",
                        help="Policy to load: ppo | carl | c3m")
    parser.add_argument("--control-scaler", type=float, default=0.1,
                        help="Scales the policy's Δu output.")
    parser.add_argument("--duration",       type=float, default=30.0,
                        help="Max audit duration in seconds.")
    parser.add_argument("--traj-path",      type=str,   default=None,
                        help="Optional path to a specific .npz trajectory. "
                             "Defaults to models/test_traj.npz via Base.load_test_data().")
    args = parser.parse_args()

    try:
        auditor = DynamicsAuditor(
            algo_name=args.algo_name,
            traj_path=args.traj_path,
        )
        auditor.run_audit(
            control_scaler=args.control_scaler,
            duration=args.duration,
        )
        auditor.stop_robot()
    except rospy.ROSInterruptException:
        pass
