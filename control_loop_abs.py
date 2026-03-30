#!/usr/bin/env python3
"""
JET-HEXA TRAJECTORY PLAYBACK (ABSOLUTE CONTROL)
----------------------------
Functionality:
1. Finds the most recent .npz data file.
2. Extracts the 18-DOF absolute joint positions from the states.
3. Publishes these absolute targets sequentially to the robot.
"""

import rospy
import numpy as np
import os
import glob
from jethexa_controller_interfaces.msg import JointCommand
from std_msgs.msg import String  # Added for safe stopping


class JetHexaTrajectoryPlayer:
    def __init__(self, directory="hexapod_data"):
        # Initialize ROS node
        rospy.init_node("jethexa_trajectory_player", anonymous=True)

        self.directory = directory

        # UPDATED: Using raw (absolute) joint control topic
        self.topic_name = "/jethexa_controller/set_joints_raw"
        self.joint_pub = rospy.Publisher(self.topic_name, JointCommand, queue_size=1)

        # Added action publisher for safely stopping the robot at the end
        self.action_pub = rospy.Publisher(
            "/jethexa_controller/run_actionset", String, queue_size=1
        )

        rospy.loginfo(f"Connecting to {self.topic_name}...")
        rospy.sleep(1.5)  # Allow time for publisher connection

    def load_latest_data(self):
        """Finds and loads the most recently modified .npz file."""
        # FIX: Changed to .npz since we are now saving a dictionary archive
        files = glob.glob(os.path.join(self.directory, "*.npz"))
        if not files:
            rospy.logerr(f"No .npz files found in directory: {self.directory}")
            return None

        latest_file = max(files, key=os.path.getmtime)
        rospy.loginfo(
            f"--- Loaded Data for Playback: {os.path.basename(latest_file)} ---"
        )
        return np.load(latest_file)

    def play_trajectory(self, hz=10.0):
        """Extracts absolute movements and streams them to the robot."""
        data = self.load_latest_data()
        if data is None:
            return

        # Extract Absolute Joint Positions (Indices 4 to 22 from 'states')
        # (Assuming your state vector still holds the physical absolute angles here)
        joint_positions = data["states"][:, 4:22]
        total_steps = len(joint_positions)

        rospy.loginfo(f"Trajectory contains {total_steps} execution steps.")

        # Setup timing
        rate = rospy.Rate(hz)
        # Command duration is set slightly shorter than the loop cycle
        step_duration = (1.0 / hz) * 0.95

        rospy.loginfo("Starting playback in 2 seconds. Ensure robot is clear!")
        rospy.sleep(2.0)

        # Execution Loop
        for i in range(total_steps):
            if rospy.is_shutdown():
                rospy.logwarn("ROS Shutdown requested. Stopping playback.")
                break

            msg = JointCommand()

            # Publish the absolute position for this timestep directly
            msg.target = joint_positions[i].tolist()
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
        """Safely stops the robot using initial_pose."""
        # CRITICAL SAFETY FIX:
        # In relative mode, sending [0.0]*18 meant "move 0 radians" (stop).
        # In absolute mode, sending [0.0]*18 means "move all servos to 0 radians",
        # which will cause the robot's legs to snap violently straight or collapse.
        # Instead, we use the bringup's safe 'initial_pose' reset.

        rospy.loginfo("Resetting robot to safe initial pose...")
        msg = String()
        msg.data = "initial_pose"
        self.action_pub.publish(msg)
        rospy.sleep(1.0)
        rospy.loginfo("Robot halted safely.")


if __name__ == "__main__":
    try:
        player = JetHexaTrajectoryPlayer(directory="hexapod_data")
        player.play_trajectory(hz=10.0)
    except rospy.ROSInterruptException:
        pass
