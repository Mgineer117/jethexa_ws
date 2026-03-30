#!/usr/bin/env python3
"""
JET-HEXA TRAJECTORY PLAYBACK
----------------------------
Functionality:
1. Finds the most recent .npy data file.
2. Extracts the 18-DOF absolute joint positions.
3. Calculates the relative deltas between each timestep.
4. Publishes these deltas sequentially to the robot.
"""

import rospy
import numpy as np
import os
import glob
from jethexa_controller_interfaces.msg import JointCommand


class JetHexaTrajectoryPlayer:
    def __init__(self, directory="hexapod_data"):
        # Initialize ROS node
        rospy.init_node("jethexa_trajectory_player", anonymous=True)

        self.directory = directory
        self.topic_name = "/jethexa_controller/set_joints_relative"
        self.joint_pub = rospy.Publisher(self.topic_name, JointCommand, queue_size=1)

        rospy.loginfo(f"Connecting to {self.topic_name}...")
        rospy.sleep(1.5)  # Allow time for publisher connection

    def load_latest_data(self):
        """Finds and loads the most recently modified .npy file."""
        files = glob.glob(os.path.join(self.directory, "*.npy"))
        if not files:
            rospy.logerr(f"No .npy files found in directory: {self.directory}")
            return None

        latest_file = max(files, key=os.path.getmtime)
        rospy.loginfo(
            f"--- Loaded Data for Playback: {os.path.basename(latest_file)} ---"
        )
        return np.load(latest_file)

    def play_trajectory(self, hz=10.0):
        """Calculates relative movements and streams them to the robot."""
        data = self.load_latest_data()
        if data is None:
            return

        # Extract only the Joint Positions (Indices 4 to 22)
        joint_controls = data["controls"]
        total_steps = len(joint_controls) - 1

        rospy.loginfo(f"Trajectory contains {total_steps} execution steps.")

        # Setup timing
        rate = rospy.Rate(hz)
        # Command duration is set slightly shorter than the loop cycle
        # to ensure smooth transitions without command queuing.
        step_duration = (1.0 / hz) * 0.95

        rospy.loginfo("Starting playback in 2 seconds. Ensure robot is clear!")
        rospy.sleep(2.0)

        # Execution Loop
        for i in range(total_steps):
            if rospy.is_shutdown():
                rospy.logwarn("ROS Shutdown requested. Stopping playback.")
                break

            msg = JointCommand()

            # Convert the numpy delta vector to a standard Python list for the ROS message
            msg.target = joint_controls[i].tolist()
            msg.duration = step_duration

            self.joint_pub.publish(msg)

            # Optional: Print progress every 20 steps
            if i % 20 == 0:
                rospy.loginfo(
                    f"Executing step {i}/{total_steps} ({(i/total_steps)*100:.1f}%)"
                )

            rate.sleep()

        rospy.loginfo("Playback sequence complete.")
        self._stop_robot()

    def _stop_robot(self):
        """Sends a zero-velocity/zero-delta command to ensure it stops."""
        stop_msg = JointCommand()
        stop_msg.target = [0.0] * 18
        stop_msg.duration = 0.5
        self.joint_pub.publish(stop_msg)
        rospy.loginfo("Robot halted.")


if __name__ == "__main__":
    try:
        # Assuming original data was recorded at 10Hz.
        # You can adjust this if you want to play it back faster/slower.
        player = JetHexaTrajectoryPlayer(directory="hexapod_data")
        player.play_trajectory(hz=10.0)
    except rospy.ROSInterruptException:
        pass
