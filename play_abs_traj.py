#!/usr/bin/env python3
"""
JET-HEXA ABSOLUTE TRAJECTORY PLAYBACK (SIMULTANEOUS 18-DOF MODE)
----------------------------
Functionality:
1. Finds the most recent .npz/.npy data file.
2. Extracts the 18-DOF absolute joint targets from the state vector.
3. Publishes all 18 joint targets simultaneously to the robot.
"""

import glob
import os

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from jethexa_controller_interfaces.msg import JointCommand
from parameters import HZ, INIT_JOINT_POS


class JetHexaAbsolutePlayer:
    def __init__(self, directory="hexapod_data"):
        rospy.init_node("jethexa_abs_player", anonymous=True)

        self.directory = directory
        self.current_joints = None

        # Subscribers
        rospy.Subscriber("/joint_states", JointState, self._joint_cb)

        # Publishers
        self.joint_pub = rospy.Publisher(
            "/jethexa_controller/set_joints_raw",
            JointCommand,
            queue_size=1,
            tcp_nodelay=True,
        )

        self.initiate_robot()

        rospy.loginfo("Waiting for /joint_states...")
        while self.current_joints is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

    def initiate_robot(self, iterations=10):
        """Forces the IK engine to engage without moving the physical legs."""
        rospy.loginfo("[INFO]: Initializing robot...")
        msg = JointCommand()
        msg.target = INIT_JOINT_POS
        msg.duration = 0.1

        for _ in range(iterations):
            if rospy.is_shutdown():
                break
            self.joint_pub.publish(msg)
            rospy.sleep(0.1)
        rospy.loginfo("[INFO]: Robot initialized.")

    def stop_robot(self):
        msg_a = JointCommand()
        msg_a.target = INIT_JOINT_POS
        msg_a.duration = 1.0
        for _ in range(5):
            if rospy.is_shutdown():
                break
            self.joint_pub.publish(msg_a)
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

    def _joint_cb(self, msg):
        self.current_joints = list(msg.position)[:18]

    def play_trajectory(self, hz=10.0):
        data_archive = self.load_latest_data()
        if data_archive is None:
            return

        """
        24-dim State Vector (Updated):
        - [0:3]   : Position (x,y,z)
        - [3:6]   : Orientation (\theta, \phi, \psi)
        - [6:24]  : Joint Positions (18 DOF)
        """
        abs_joint_targets = data_archive["states"][:, 6:24]
        total_steps = len(abs_joint_targets)

        # In non-gait mode, we run at standard Hz
        rate = rospy.Rate(hz)
        step_duration = (1.0 / hz) * 0.95

        # Execution Loop
        rospy.loginfo(f"[INFO]: Playing Trajectory with {total_steps} execution steps.")
        for i in range(total_steps):
            if rospy.is_shutdown():
                break

            target_full = abs_joint_targets[i].tolist()

            # --- Move ALL 18 joints simultaneously ---
            msg = JointCommand()
            msg.target = target_full
            msg.duration = step_duration
            self.joint_pub.publish(msg)
            rate.sleep()

            if i % 10 == 0:
                rospy.loginfo(
                    f"[INFO]: Executing step {(i/total_steps)*100:.1f}% Complete."
                )

        rospy.loginfo("[INFO]: Absolute Playback complete.")


if __name__ == "__main__":
    try:
        player = JetHexaAbsolutePlayer()
        player.play_trajectory(hz=HZ)
        player.stop_robot()
    except rospy.ROSInterruptException:
        pass
