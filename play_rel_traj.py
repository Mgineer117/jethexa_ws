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

import numpy as np
import rospy

from base import Base
from jethexa_controller_interfaces.msg import JointCommand

config = {
    "use_mocap": False,
    "use_imu": False,
    "listen_joint_states": True,
    "use_abs_joint_commands": True,
    "use_rel_joint_commands": False,
    "load_test_traj": True,
    "load_recent_traj": False,
}


class JetHexaTrajectoryPlayer(Base):
    def __init__(self):
        rospy.init_node("jethexa_trajectory_player", anonymous=True)
        super().__init__(config=config)

    def play_trajectory(self):
        joint_angles = self.test_data["states"][:, 6:24]
        joint_controls = self.test_data["controls"]
        joint_deltas = joint_controls * self.dt
        total_steps = len(joint_controls)

        # Start exactly at states[0], matching the plot
        target_joint_pos = np.array(joint_angles[0], dtype=float)

        rospy.loginfo(f"[INFO]: Playing Trajectory with {total_steps} execution steps.")
        for i in range(total_steps):
            if rospy.is_shutdown():
                break

            # 2. Then integrate using controls[i+1] for next step
            target_joint_pos += joint_deltas[i]

            # 1. Publish current target FIRST (at i=0, publishes states[0])
            msg = JointCommand()
            msg.target = target_joint_pos.tolist()
            msg.duration = self.duration

            # Keep publishing and checking until the error is within the threshold
            j = 0
            while not rospy.is_shutdown():
                self.joint_pub.publish(msg)
                self.rate.sleep()

                sq_error = np.sum((self.joint_pos - target_joint_pos) ** 2)
                if j % 5 == 0:  # Log every 5 iterations to avoid spamming
                    rospy.loginfo(
                        f"[INFO] Moving robot {j}... Squared joint error: {sq_error:.6f}"
                    )
                if sq_error < self.joint_error_threshold:
                    break
                j += 1

            if i % 10 == 0:
                rospy.loginfo(
                    f"[INFO]: Executing step {(i/total_steps)*100:.1f}% Complete."
                )

        rospy.loginfo("[INFO]: Integrated Virtual Absolute Playback complete.")
        self.stop_robot()


if __name__ == "__main__":
    try:
        player = JetHexaTrajectoryPlayer()
        player.initialize_robot_for_replay()
        player.play_trajectory()
        player.stop_robot()
    except rospy.ROSInterruptException:
        pass
