#!/usr/bin/env python3
import argparse
import os
import time

import numpy as np
import rospy
import torch
import torch.nn as nn
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Imu, JointState

from jethexa_controller_interfaces.msg import JointCommand
from models.policy_networks import CLActor


class JetHexaRLCollector:
    def __init__(self, algo_name, hz=10.0):
        rospy.init_node("jethexa_rl_collector", anonymous=True)

        self.output_dir = "hexapod_data"
        self.ref_dir = "models/test_traj.npz"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Timing and Rates
        self.hz = hz
        self.dt = 1.0 / self.hz
        self.rate = rospy.Rate(self.hz)

        # Data Buffers
        self.recorded_states = []
        self.recorded_actions = []

        # Robot State Variables
        self.base_pos = np.array([0.0, 0.0])
        self.base_attitude = np.array([0.0, 0.0, 0.0])
        self.joint_pos = np.zeros(18)
        self.prev_joint_pos = np.zeros(18)

        self.joints_ready = False
        self.mocap_ready = False

        # Load PyTorch Model
        self.policy = self.load_policy(algo_name)

        # ROS Communication
        self.joint_pub = rospy.Publisher(
            "/jethexa_controller/set_joints_relative",
            JointCommand,
            queue_size=1,
            tcp_nodelay=True,
        )
        rospy.Subscriber("/joint_states", JointState, self.joint_cb)
        rospy.Subscriber("/qualysis/jethexa", PoseStamped, self.base_pos_cb)

        # --- WARMUP SEQUENCE ---
        self.warmup_controller()

    def base_pos_cb(self, msg):
        self.base_pos = np.array([msg.pose.position.x, msg.pose.position.y])
        self.base_attitude = np.array(
            [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]
        )
        self.mocap_ready = True

    def joint_cb(self, msg):
        self.joint_pos = np.array(msg.position)
        self.joints_ready = True

    def load_policy(self, algo_name):
        if algo_name == "ppo":
            path = "models/ppo.pth"
            mode = "deterministic"
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

    def stop_robot(self):
        msg = JointCommand()
        msg.target = [0.0] * 18
        msg.duration = 0.5
        self.joint_pub.publish(msg)
        rospy.loginfo("Robot locked at current pose.")

    def warmup_controller(self):
        """Primes the IK engine by sending current-state heartbeats."""
        rospy.loginfo("Waiting for sensor stream...")
        while not (self.joints_ready and self.mocap_ready) and not rospy.is_shutdown():
            rospy.sleep(0.1)

        rospy.loginfo("Priming controller with 1.5s heartbeat...")
        msg = JointCommand()
        msg.target = [0.0] * 18  # 0 delta = stay where you are
        msg.duration = self.dt

        for _ in range(int(1.5 * self.hz)):
            self.joint_pub.publish(msg)
            rospy.sleep(self.dt)

        self.prev_joint_pos = np.copy(self.joint_pos)
        rospy.loginfo("Warmup complete. Controller is 'hot'.")

    def get_action_from_policy(self, state):
        """
        Policy outputs Joint Velocity (rad/s).
        We return relative delta (rad) = Velocity * dt
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            # action_rate = self.policy(state_tensor).cpu().numpy().squeeze()
            action_rate = np.random.uniform(-0.5, 0.5, 18)  # Mock inference

        # Scale by dt to get the relative displacement for this timestep
        return action_rate * self.dt

    def run_rollout(self, duration=33.0):
        self.recorded_states = []
        self.recorded_actions = []
        start_time = rospy.get_time()

        xref = np.load(self.ref_dir)["states"]
        xref = np.delete(xref, 2, axis=1)
        uref = np.load(self.ref_dir)["controls"]

        rospy.loginfo("Starting Policy Rollout...")

        while not rospy.is_shutdown():
            elapsed = rospy.get_time() - start_time
            if elapsed >= duration:
                break

            # NEW: Calculate index dynamically based on actual time
            i = int(elapsed / self.dt)

            # NEW: Safety catch to prevent IndexError
            if i >= len(xref):
                rospy.logwarn(
                    "Reached end of reference trajectory. Ending rollout early."
                )
                break

            # 1. Get current state and policy decision
            x = np.concatenate([self.base_pos, self.base_attitude, self.joint_pos])
            state = np.concatenate([x, xref[i], uref[i]])

            with torch.no_grad():
                du, _ = self.policy(torch.from_numpy(state).float().unsqueeze(0))
            u = uref[i] + du.cpu().numpy().squeeze()

            target = u * self.dt

            # 2. Command the robot
            msg = JointCommand()
            msg.target = target.tolist()
            msg.duration = self.dt * 0.9
            self.joint_pub.publish(msg)

            # 3. Record
            self.recorded_states.append(state)
            self.recorded_actions.append(u)

            # 4. Prepare for next iteration
            self.prev_joint_pos = np.copy(self.joint_pos)
            self.rate.sleep()

        self.stop_robot()
        self.save_data()

    def save_data(self):
        if not self.recorded_states:
            return
        ts = int(time.time())
        filename = os.path.join(self.output_dir, f"rl_policy_rollout_{ts}.npz")
        np.savez_compressed(
            filename,
            states=np.array(self.recorded_states),
            actions=np.array(self.recorded_actions),
        )
        rospy.loginfo(f"Data saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo-name", type=str, default="ppo")
    parser.add_argument("--duration", type=float, default=33.0)
    args = parser.parse_args()

    try:
        collector = JetHexaRLCollector(algo_name=args.algo_name, hz=10.0)
        collector.run_rollout(duration=args.duration)
    except rospy.ROSInterruptException:
        pass
