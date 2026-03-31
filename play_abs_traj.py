#!/usr/bin/env python3
import glob
import os

import numpy as np
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from jethexa_controller_interfaces.msg import JointCommand


class JetHexaAbsolutePlayer:
    def __init__(self, directory="hexapod_data"):
        rospy.init_node("jethexa_abs_player", anonymous=True)

        self.directory = directory

        # CORRECTED: Using the raw absolute joint control topic
        self.topic_name = "/jethexa_controller/set_joints_raw"

        # Dictionary for Tripod Grouping
        self.groups = {
            "Group A": [0, 1, 2, 6, 7, 8, 12, 13, 14],
            "Group B": [3, 4, 5, 9, 10, 11, 15, 16, 17],
        }

        self.init_joint_pos = [
            0.17120142291266816,
            0.7383813588273885,
            -0.574647577430417,
            0.0,
            0.7435823754497041,
            -0.5952505177281151,
            -0.17120142291266816,
            0.7383813588273885,
            -0.574647577430417,
            0.17120142291266838,
            0.7383813588273884,
            -0.5746475774304161,
            0.0,
            0.7435823754497041,
            -0.5952505177281151,
            -0.17120142291266838,
            0.7383813588273884,
            -0.5746475774304161,
        ]

        self.current_joints = None

        rospy.Subscriber("/joint_states", JointState, self._joint_cb)
        self.joint_pub = rospy.Publisher(
            self.topic_name, JointCommand, queue_size=1, tcp_nodelay=True
        )

        rospy.loginfo("Waiting for /joint_states...")
        while self.current_joints is None and not rospy.is_shutdown():
            rospy.sleep(0.1)

        rospy.loginfo("Connection established. Priming controller...")
        self.wake_up_with_current_state()

    def _joint_cb(self, msg):
        self.current_joints = list(msg.position)[:18]

    def wake_up_with_current_state(self, iterations=15):
        """Forces the IK engine to engage without moving the physical legs."""
        rospy.loginfo("Sending current state heartbeat...")
        msg = JointCommand()
        msg.target = self.init_joint_pos
        msg.duration = 0.1

        for _ in range(iterations):
            if rospy.is_shutdown():
                break
            self.joint_pub.publish(msg)
            rospy.sleep(0.05)
        rospy.loginfo("Controller warmed up and ready.")

    def stop_robot(self):
        msg_a = JointCommand()
        msg_a.target = self.init_joint_pos
        msg_a.duration = 1.0
        for _ in range(5):
            if rospy.is_shutdown():
                break
            self.joint_pub.publish(msg_a)

    def load_latest_data(self):
        files = glob.glob(os.path.join(self.directory, "*.npz"))
        if not files:
            rospy.logerr(f"No .npz files found in: {self.directory}")
            return None
        latest_file = max(files, key=os.path.getmtime)
        return np.load(latest_file)

    def play_trajectory(self, hz=10.0):
        data_archive = self.load_latest_data()
        if data_archive is None:
            return

        # Extract the 18 joint angles from the state vector
        abs_joint_targets = data_archive["states"][:, 4:22]
        total_steps = len(abs_joint_targets)

        # Run at 2x speed to accommodate split phases
        rate = rospy.Rate(hz * 2.0)
        step_duration = (1.0 / (hz * 2.0)) * 0.95

        rospy.loginfo("Starting Absolute Tripod Playback...")

        # Track the "last known" commanded positions to use as freeze points
        last_commanded = self.current_joints.copy()

        for i in range(total_steps):
            if rospy.is_shutdown():
                break

            target_full = abs_joint_targets[i].tolist()

            # --- PHASE 1: Move Tripod A, Freeze Tripod B ---
            msg_a = JointCommand()
            # If joint is in Group A, update to the new target.
            # If in Group B, hold at the last commanded position.
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
            # Now update Group B to the new target.
            # Group A remains held at its current position.
            msg_b.target = [
                target_full[j] if j in self.groups["Group B"] else last_commanded[j]
                for j in range(18)
            ]
            msg_b.duration = step_duration
            self.joint_pub.publish(msg_b)
            last_commanded = msg_b.target.copy()
            rate.sleep()

        rospy.loginfo("Playback complete.")
        self.stop_robot()


if __name__ == "__main__":
    try:
        player = JetHexaAbsolutePlayer()
        player.play_trajectory(hz=10.0)
        player.stop_robot()
    except rospy.ROSInterruptException:
        pass
