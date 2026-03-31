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


class JetHexaTrajectoryPlayer:
    def __init__(self, directory="hexapod_data"):
        rospy.init_node("jethexa_trajectory_player", anonymous=True)

        self.directory = directory

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

        self.joint_rel_pub = rospy.Publisher(
            "/jethexa_controller/set_joints_relative", JointCommand, queue_size=1
        )
        self.joint_abs_pub = rospy.Publisher(
            "/jethexa_controller/set_joints_raw",
            JointCommand,
            queue_size=1,
            tcp_nodelay=True,
        )

        rospy.sleep(1.5)

        self.wake_up_with_current_state()

    def wake_up_with_current_state(self, iterations=15):
        """Forces the IK engine to engage without moving the physical legs."""
        rospy.loginfo("Sending current state heartbeat...")
        msg = JointCommand()
        msg.target = self.init_joint_pos
        msg.duration = 0.1

        for _ in range(iterations):
            if rospy.is_shutdown():
                break
            self.joint_abs_pub.publish(msg)
            rospy.sleep(0.05)
        rospy.loginfo("Controller warmed up and ready.")

    def stop_robot(self):
        msg_a = JointCommand()
        msg_a.target = self.init_joint_pos
        msg_a.duration = 1.0
        for _ in range(5):
            if rospy.is_shutdown():
                break
            self.joint_abs_pub.publish(msg_a)

    def load_latest_data(self):
        files = glob.glob(os.path.join(self.directory, "*.npz"))
        if not files:
            rospy.logerr(f"No .npz files found in: {self.directory}")
            return None
        latest_file = max(files, key=os.path.getmtime)
        return np.load(latest_file)

    def play_trajectory(self, hz=10.0):
        data = self.load_latest_data()
        if data is None:
            return

        joint_controls = data["controls"]
        total_steps = len(joint_controls) - 1

        rospy.loginfo(f"Trajectory contains {total_steps} execution steps.")

        # In tripod mode, we do TWO movements per timestep (Group A, then Group B).
        # We run the loop twice as fast so the overall time stays the same.
        rate = rospy.Rate(hz * 2.0)
        step_duration = (1.0 / (hz * 2.0)) * 0.95

        rospy.loginfo("Starting Tripod playback in 2 seconds...")
        rospy.sleep(2.0)

        # Execution Loop
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
                    f"Executing step {i}/{total_steps} ({(i/total_steps)*100:.1f}%)"
                )

        rospy.loginfo("Playback sequence complete.")
        self.stop_robot()


if __name__ == "__main__":
    try:
        player = JetHexaTrajectoryPlayer(directory="hexapod_data")
        player.play_trajectory(hz=10.0)
    except rospy.ROSInterruptException:
        pass
