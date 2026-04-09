#!/usr/bin/env python3
"""
JET-HEXA TRAJECTORY PLAYBACK (INTEGRATED VIRTUAL ABSOLUTE MODE)
----------------------------
Functionality:
1. Finds the most recent .npz/.npy data file.
2. Extracts the 18-DOF joint controls (velocities/deltas).
3. Integrates controls from the initial position to calculate virtual absolute targets.
4. Publishes all 18 absolute joint targets simultaneously to the robot.
"""

import glob
import os

import numpy as np
import rospy

from jethexa_controller_interfaces.msg import JointCommand
from parameters import HZ, INIT_JOINT_POS


class JetHexaTrajectoryPlayer:
    def __init__(self, directory="hexapod_data", hz=10.0):
        rospy.init_node("jethexa_trajectory_player", anonymous=True)

        self.directory = directory
        self.hz = hz
        self.dt = 1.0 / self.hz
        self.rate = rospy.Rate(self.hz)

        # Publisher: We only need the absolute (raw) publisher now
        self.joint_abs_pub = rospy.Publisher(
            "/jethexa_controller/set_joints_raw",
            JointCommand,
            queue_size=1,
            tcp_nodelay=True,
        )

        self.data = self.load_test_data()
        if self.data is None:
            rospy.logerr("No trajectory data found. Exiting.")
            exit(1)

        self.init_ref_joint_pos = self.data["states"][0, 6:24].tolist()

        self.initiate_robot()

    def initiate_robot(self, iterations=10):
        """Forces the IK engine to engage without moving the physical legs."""
        rospy.loginfo("[INFO]: Initializing robot...")

        msg = JointCommand()
        msg.target = self.init_ref_joint_pos
        msg.duration = 0.1

        for _ in range(iterations):
            if rospy.is_shutdown():
                break
            self.joint_abs_pub.publish(msg)
            rospy.sleep(0.1)
        rospy.loginfo("[INFO]: Robot initialized.")

    def stop_robot(self):
        rospy.loginfo("[INFO]: Stopping robot and returning to home...")
        msg_a = JointCommand()
        msg_a.target = INIT_JOINT_POS
        msg_a.duration = 1.0
        for _ in range(5):
            if rospy.is_shutdown():
                break
            self.joint_abs_pub.publish(msg_a)
            rospy.sleep(0.1)

    def load_latest_data(self, filename=None):
        """
        24-dim State Vector (Updated):
        - [0:3]   : Position (x,y,z)
        - [3:6]   : Orientation (\theta, \phi, \psi)
        - [6:24]  : Joint Positions (18 DOF)
        """
        if filename:
            return np.load(os.path.join(self.directory, filename))

        files = glob.glob(os.path.join(self.directory, "*.npz"))
        if not files:
            rospy.logerr(f"No .npz files found in: {self.directory}")
            return None
        latest_file = max(files, key=os.path.getmtime)
        rospy.loginfo(f"[INFO]: Loading data from {latest_file}")
        return np.load(latest_file)

    def load_test_data(self):
        """
        24-dim State Vector (Updated):
        - [0:3]   : Position (x,y,z)
        - [3:6]   : Orientation (\theta, \phi, \psi)
        - [6:24]  : Joint Positions (18 DOF)
        """
        test_file = os.path.join("models/test_traj.npz")
        if not os.path.exists(test_file):
            rospy.logerr(f"Test file not found: {test_file}")
            return None
        return np.load(test_file)

    def play_trajectory(self):
        joint_angles = self.data["states"][:, 6:24]
        joint_controls = self.data["controls"]
        joint_deltas = joint_controls * self.dt
        total_steps = len(joint_controls)

        step_duration = self.dt * 0.95

        # Start exactly at states[0], matching the plot
        virtual_target = np.array(self.init_ref_joint_pos, dtype=float)

        rospy.loginfo(f"[INFO]: Playing Trajectory with {total_steps} execution steps.")
        for i in range(total_steps):
            if rospy.is_shutdown():
                break

            # 2. Then integrate using controls[i+1] for next step
            virtual_target = virtual_target + joint_deltas[i]

            # 1. Publish current target FIRST (at i=0, publishes states[0])
            msg = JointCommand()
            msg.target = virtual_target.tolist()
            msg.duration = step_duration
            self.joint_abs_pub.publish(msg)
            self.rate.sleep()

            if i % 10 == 0:
                error = np.linalg.norm(virtual_target - joint_angles[i + 1])
                rospy.loginfo(
                    f"[INFO]: Executing step {(i/total_steps)*100:.1f}% Complete. Error: {error:.4f}."
                )

        rospy.loginfo("[INFO]: Integrated Virtual Absolute Playback complete.")
        self.stop_robot()


if __name__ == "__main__":
    try:
        player = JetHexaTrajectoryPlayer(directory="hexapod_data", hz=HZ)
        player.play_trajectory()
    except rospy.ROSInterruptException:
        pass
