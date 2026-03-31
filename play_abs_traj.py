#!/usr/bin/env python3
import glob
import os

import numpy as np
from parameters import GROUP_A, GROUP_B, HZ, INIT_JOINT_POS
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from jethexa_controller_interfaces.msg import JointCommand


class JetHexaAbsolutePlayer:
    def __init__(self, directory="hexapod_data"):
        rospy.init_node("jethexa_abs_player", anonymous=True)

        self.directory = directory

        # Dictionary for Tripod Grouping
        self.groups = {
            "Group A": GROUP_A,
            "Group B": GROUP_B,
        }

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
            self.joint_pub.publish(msg_a)

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

        # Extract the 18 joint angles from the state vector
        """
        40-dim State Vector:
        - [0:3]   : Position (x,y,z)
        - [3:7]   : Quat (x,y,z,w)
        - [7:25]  : Joint Positions (rad)
        - [25:43] : Joint Velocities (rad/s)
        """
        abs_joint_targets = data_archive["states"][:, 7:25]
        total_steps = len(abs_joint_targets)

        # Run at 2x speed to accommodate split phases
        rate = rospy.Rate(hz * 2.0)
        step_duration = (1.0 / (hz * 2.0)) * 0.95

        # Track the "last known" commanded positions to use as freeze points
        last_commanded = self.current_joints.copy()

        # Execution Loop
        rospy.loginfo(
            f"[INFO]: Collecting Trajectory with {total_steps} execution steps."
        )
        for i in range(total_steps):
            if rospy.is_shutdown():
                break

            target_full = abs_joint_targets[i].tolist()

            # --- PHASE 1: Move Tripod A, Freeze Tripod B ---
            msg_a = JointCommand()
            msg_a.target = [
                target_full[j] if j in self.groups["Group A"] else last_commanded[j]
                for j in range(18)
            ]
            msg_a.duration = step_duration
            self.joint_pub.publish(msg_a)
            last_commanded = msg_a.target.copy()
            rate.sleep()

            # --- PHASE 2: Move Tripod B, Freeze Tripod A ---
            msg_b = JointCommand()
            msg_b.target = [
                target_full[j] if j in self.groups["Group B"] else last_commanded[j]
                for j in range(18)
            ]
            msg_b.duration = step_duration
            self.joint_pub.publish(msg_b)
            last_commanded = msg_b.target.copy()
            rate.sleep()

            if i % 10 == 0:
                rospy.loginfo(
                    f"[INFO]: Executing step {(i/total_steps)*100:.1f}% Complete."
                )

        rospy.loginfo("[INFO]: Absolute Tripod Playback complete.")
        self.stop_robot()


if __name__ == "__main__":
    try:
        player = JetHexaAbsolutePlayer()
        player.play_trajectory(hz=HZ)
        player.stop_robot()
    except rospy.ROSInterruptException:
        pass
