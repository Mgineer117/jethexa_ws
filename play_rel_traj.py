#!/usr/bin/env python3
"""
JET-HEXA TRAJECTORY PLAYBACK (SIMULTANEOUS 18-DOF MODE)
----------------------------
Functionality:
1. Finds the most recent .npz/.npy data file.
2. Extracts the 18-DOF relative joint controls.
3. Publishes all 18 joint deltas simultaneously to the robot.
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

    def play_trajectory(self):
        # data = self.load_latest_data()
        data = self.load_test_data()
        if data is None:
            return

        joint_controls = data["controls"]
        joint_targets = joint_controls * self.dt
        total_steps = len(joint_controls)

        # In non-gait mode, we do ONE movement per timestep containing all 18 DOF.
        step_duration = self.dt * 0.95  # add margin

        # Execution Loop
        rospy.loginfo(f"[INFO]: Playing Trajectory with {total_steps} execution steps.")
        for i in range(total_steps):
            if rospy.is_shutdown():
                break

            target_full = joint_targets[i].tolist()

            # --- Move ALL 18 joints simultaneously ---
            msg = JointCommand()
            msg.target = target_full
            msg.duration = step_duration
            self.joint_rel_pub.publish(msg)
            self.rate.sleep()

            if i % 10 == 0:
                rospy.loginfo(
                    f"[INFO]: Executing step {(i/total_steps)*100:.1f}% Complete."
                )

        rospy.loginfo("[INFO]: Relative Playback complete.")
        self.stop_robot()


if __name__ == "__main__":
    try:
        player = JetHexaTrajectoryPlayer(directory="hexapod_data", hz=HZ)
        player.play_trajectory()
    except rospy.ROSInterruptException:
        pass
