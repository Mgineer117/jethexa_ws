#!/usr/bin/env python3
"""
JET-HEXA TRAJECTORY PLAYBACK (FORCED TRIPOD MODE)
----------------------------
Functionality:
1. Finds the most recent .npz/.npy data file.
2. Extracts the 18-DOF relative joint controls.
3. Splits each step into Tripod A and Tripod B movements.
4. Publishes these deltas alternately to the robot.
"""

import glob
import os

import numpy as np
import rospy

from jethexa_controller_interfaces.msg import JointCommand
from parameters import GROUP_A, GROUP_B, HZ, INIT_JOINT_POS


class JetHexaTrajectoryPlayer:
    def __init__(self, directory="hexapod_data"):
        rospy.init_node("jethexa_trajectory_player", anonymous=True)

        self.directory = directory

        self.groups = {
            "Group A": GROUP_A,
            "Group B": GROUP_B,
        }

        # Publishers
        self.joint_rel_pub = rospy.Publisher(
            "/jethexa_controller/set_joints_relative", JointCommand, queue_size=1
        )
        self.joint_abs_pub = rospy.Publisher(
            "/jethexa_controller/set_joints_raw",
            JointCommand,
            queue_size=1,
            tcp_nodelay=True,
        )

        self.initiate_robot()

    def initiate_robot(self, iterations=10):
        """Forces the IK engine to engage without moving the physical legs."""
        rospy.loginfo("[INFO]: Initializing robot...")
        msg = JointCommand()
        msg.target = INIT_JOINT_POS
        msg.duration = 0.1

        for _ in range(iterations):
            if rospy.is_shutdown():
                break
            self.joint_abs_pub.publish(msg)
            rospy.sleep(0.1)
        rospy.loginfo("[INFO]: Robot initialized.")

    def stop_robot(self):
        msg_a = JointCommand()
        msg_a.target = INIT_JOINT_POS
        msg_a.duration = 1.0
        for _ in range(5):
            if rospy.is_shutdown():
                break
            self.joint_abs_pub.publish(msg_a)
            rospy.sleep(0.1)

    def load_latest_data(self, filename=None):
        if filename:
            return np.load(os.path.join(self.directory, filename))

        files = glob.glob(os.path.join(self.directory, "*.npz"))
        if not files:
            rospy.logerr(f"No .npz files found in: {self.directory}")
            return None
        latest_file = max(files, key=os.path.getmtime)
        return np.load(latest_file)

    def load_test_data(self):
        test_file = os.path.join("models/test_traj.npz")
        if not os.path.exists(test_file):
            rospy.logerr(f"Test file not found: {test_file}")
            return None
        return np.load(test_file)

    def play_trajectory(self, hz=10.0):
        # data = self.load_latest_data()
        data = self.load_test_data()
        if data is None:
            return

        joint_controls = data["controls"]
        total_steps = len(joint_controls)

        # In tripod mode, we do TWO movements per timestep (Group A, then Group B).
        rate = rospy.Rate(hz * 2.0)
        step_duration = (1.0 / (hz * 2.0)) * 0.95  # add margin

        # Execution Loop
        rospy.loginfo(
            f"[INFO]: Playing Trajectory with {total_steps} execution steps."  # BUG FIX: Changed Collecting to Playing
        )
        for i in range(total_steps):
            if rospy.is_shutdown():
                break

            target_full = joint_controls[i].tolist()

            # --- PHASE 1: Move Tripod A, Freeze Tripod B ---
            msg_a = JointCommand()
            msg_a.target = [
                target_full[j] if j in self.groups["Group A"] else 0.0
                for j in range(18)
            ]
            msg_a.duration = step_duration
            self.joint_rel_pub.publish(msg_a)
            rate.sleep()

            # --- PHASE 2: Move Tripod B, Freeze Tripod A ---
            msg_b = JointCommand()
            msg_b.target = [
                target_full[j] if j in self.groups["Group B"] else 0.0
                for j in range(18)
            ]
            msg_b.duration = step_duration
            self.joint_rel_pub.publish(msg_b)
            rate.sleep()

            if i % 10 == 0:
                rospy.loginfo(
                    f"[INFO]: Executing step {(i/total_steps)*100:.1f}% Complete."
                )

        rospy.loginfo("[INFO]: Relative Tripod Playback complete.")
        self.stop_robot()


if __name__ == "__main__":
    try:
        player = JetHexaTrajectoryPlayer(directory="hexapod_data")
        player.play_trajectory(hz=HZ)
    except rospy.ROSInterruptException:
        pass
