#!/usr/bin/env python3
import argparse
import os
import time

import numpy as np
import rospy
import torch

from base import Base
from jethexa_controller_interfaces.msg import JointCommand
from models.policy_networks import CLActor

config = {
    "use_mocap": True,
    "use_imu": False,
    "listen_joint_states": True,
    "use_abs_joint_commands": True,
    "use_rel_joint_commands": False,
    "load_test_traj": True,
    "load_recent_traj": False,
}


class JetHexaRLCollector(Base):
    def __init__(self, algo_name):
        rospy.init_node("jethexa_rl_collector", anonymous=True)

        super().__init__(config=config)

        # Data Buffers
        self.recorded_states = []
        self.recorded_actions = []

        # Load PyTorch Model
        self.policy = self.load_policy(algo_name)

    def load_policy(self, algo_name):
        if algo_name == "ppo":
            path = "models/ppo.pth"
            mode = "stochastic"
        elif algo_name == "carl":
            path = "models/carl.pth"
            mode = "deterministic"
        elif algo_name == "c3m":
            path = "models/c3m.pth"
            mode = "deterministic"
        else:
            raise ValueError(
                f"Unknown algorithm: {algo_name}. Choose ppo, carl, or c3m."
            )

        actor = CLActor(
            x_dim=23,
            u_dim=18,
            num_windows=1,
            mode=mode,
        )
        actor.load_state_dict(torch.load(path, map_location=torch.device("cpu")))
        actor.eval()

        rospy.loginfo(f"Loading policy from {path}...")

        return actor

    def run_rollout(self, control_scaler: float, duration=33.0):
        self.recorded_states, self.recorded_actions = [], []

        xref = np.delete(self.test_data["states"], 2, axis=1)[-300:, :]
        uref = self.test_data["controls"][-300:, :]

        total_steps = min(len(xref), int(duration / self.dt))

        rospy.loginfo(f"[INFO]: Playing Trajectory with {total_steps} execution steps.")
        for i in range(total_steps):
            if rospy.is_shutdown():
                break

            # 1. Get current state and policy decision
            base_pos, base_attitude, joint_pos = (
                self.base_pos,
                self.base_attitude,
                self.joint_pos,
            )
            x = np.concatenate([base_pos, base_attitude, joint_pos])

            state = np.concatenate([x, xref[i], uref[i]])
            with torch.no_grad():
                du, _ = self.policy(torch.from_numpy(state).float().unsqueeze(0))
                if torch.isnan(du).any():
                    rospy.logwarn(
                        f"NaN detected in policy output at step {i}. Using zero action."
                    )
                    du = torch.zeros_like(du)
                du = control_scaler * du.cpu().numpy().squeeze()

            u = uref[i] + du
            u = np.clip(u, -4.0, 4.0)

            # delta = u * self.dt
            target_joint_pos = joint_pos + u * self.dt
            target_joint_pos = self.check_valid_joint_angle(
                target_joint_pos, terminate_on_invalid=False
            )

            # 1. Publish current target FIRST (at i=0, publishes states[0])
            msg = JointCommand()
            msg.target = target_joint_pos.tolist()
            msg.duration = self.duration

            self.joint_abs_pub.publish(msg)
            self.rate.sleep()

            # 3. Record (rest of your code)
            self.recorded_states.append(state)
            self.recorded_actions.append(u)

        self.save_data()

    def save_data(self):
        if not self.recorded_states:
            return
        ts = int(time.time())
        filename = os.path.join("hexapod_data", f"rl_policy_rollout_{ts}.npz")
        np.savez_compressed(
            filename,
            states=np.array(self.recorded_states),
            actions=np.array(self.recorded_actions),
        )
        rospy.loginfo(f"Data saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="ppo")
    parser.add_argument("--control-scaler", type=float, default=0.3)
    parser.add_argument("--duration", type=float, default=33.0)
    args = parser.parse_args()

    try:
        collector = JetHexaRLCollector(algo_name=args.algo_name)
        collector.initialize_robot_for_replay()
        collector.run_rollout(
            control_scaler=args.control_scaler, duration=args.duration
        )
        collector.stop_robot()
    except rospy.ROSInterruptException:
        pass
