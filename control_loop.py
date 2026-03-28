#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import JointState, Imu
import time
import os
import argparse
import subprocess


class JetHexaDataCollector:
    def __init__(self, hz=25.0):
        rospy.init_node("jethexa_data_collector", anonymous=True)

        self.output_dir = "hexapod_rl_data"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.hz = hz
        self.dt = 1.0 / self.hz
        self.rate = rospy.Rate(self.hz)

        self.recorded_states = []
        self.recorded_actions = []

        self.joint_pos = np.zeros(18)
        self.prev_joint_pos = np.zeros(18)
        self.base_quat = np.array([0.0, 0.0, 0.0, 1.0])

        rospy.Subscriber("/joint_states", JointState, self.joint_cb)
        rospy.Subscriber("/imu/filtered", Imu, self.imu_cb)

        rospy.loginfo("Waiting for sensors...")
        rospy.sleep(2.0)
        self.prev_joint_pos = np.copy(self.joint_pos)

    def joint_cb(self, msg):
        self.joint_pos = np.array(msg.position)

    def imu_cb(self, msg):
        q = msg.orientation
        self.base_quat = np.array([q.x, q.y, q.z, q.w])

    def get_current_state(self):
        joint_vel = (self.joint_pos - self.prev_joint_pos) / self.dt
        state = np.concatenate([self.base_quat, self.joint_pos, joint_vel])
        self.prev_joint_pos = np.copy(self.joint_pos)
        return state

    def execute_bash_command(self, action_deltas):
        """
        Constructs and executes the rostopic pub bash command.
        """
        # 1. Format the 18 joints + 1 duration into a comma-separated string
        # e.g., "-0.1, 0.1, 0.0, ..., 0.1"
        action_list = action_deltas.tolist() + [self.dt]
        data_str = ", ".join(map(str, action_list))

        # 2. Build the exact bash string you requested
        bash_cmd = f'rostopic pub -1 /jethexa_controller/set_joints_relative std_msgs/Float32MultiArray "{{data: [{data_str}]}}"'

        # 3. Execute it in the background using subprocess
        # We use Popen instead of run() so it doesn't completely freeze the Python script
        # while waiting for the 3-second ROS latching timeout.
        subprocess.Popen(
            bash_cmd,
            shell=True,
            executable="/bin/bash",
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def stop_robot(self):
        stop_action = np.zeros(18)
        self.execute_bash_command(stop_action)
        rospy.sleep(1.0)

    def _run_collection(self, mode, duration, get_action_func):
        self.recorded_states = []
        self.recorded_actions = []
        start_time = rospy.get_time()

        rospy.loginfo(f"Starting {mode} RL collection using Bash Subprocess...")

        while not rospy.is_shutdown() and (rospy.get_time() - start_time) < duration:
            elapsed = rospy.get_time() - start_time

            # 1. Get the 18-dim action
            action_deltas = get_action_func(elapsed)

            # 2. Fire the Bash Command
            self.execute_bash_command(action_deltas)

            # 3. Record State and Action
            self.recorded_states.append(self.get_current_state())
            self.recorded_actions.append(action_deltas)

            self.rate.sleep()

        self.stop_robot()
        self.save_data(mode)

    def collect_random_uniform(self, duration=10.0):
        def action_logic(t):
            return np.random.uniform(-0.1, 0.1, size=18)

        self._run_collection("random_uniform", duration, action_logic)

    def save_data(self, mode):
        if not self.recorded_states:
            return
        ts = int(time.time())
        filename = os.path.join(self.output_dir, f"rl_rollout_{mode}_{ts}.npz")
        np.savez_compressed(
            filename,
            states=np.array(self.recorded_states),
            actions=np.array(self.recorded_actions),
        )
        rospy.loginfo(f"Saved {mode} rollouts to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["random"], default="random")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--hz", type=float, default=25.0)
    args = parser.parse_args()

    try:
        collector = JetHexaDataCollector(hz=args.hz)
        collector.collect_random_uniform(duration=args.duration)
    except rospy.ROSInterruptException:
        pass
