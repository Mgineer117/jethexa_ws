#!/usr/bin/env python3
"""
JET-HEXA ABSOLUTE TRAJECTORY PLAYBACK (SIMULTANEOUS 18-DOF MODE)
----------------------------
Functionality:
1. Finds the most recent .npz/.npy data file.
2. Extracts the 18-DOF absolute joint targets from the state vector.
3. Publishes all 18 joint targets simultaneously to the robot.
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


class JetHexaAbsolutePlayer(Base):
    def __init__(
        self,
    ):
        super().__init__(config=config)

        rospy.init_node("jethexa_abs_player", anonymous=True)

    def play_trajectory(self):
        abs_joint_targets = self.test_data["states"][:, 6:24]
        total_steps = len(abs_joint_targets)

        # Execution Loop
        rospy.loginfo(f"[INFO]: Playing Trajectory with {total_steps} execution steps.")
        for i in range(total_steps):
            if rospy.is_shutdown():
                break

            target_joint_pos = abs_joint_targets[i].tolist()

            # --- Move ALL 18 joints simultaneously ---
            msg = JointCommand()
            msg.target = target_joint_pos
            msg.duration = self.duration

            # Keep publishing and checking until the error is within the threshold
            while not rospy.is_shutdown():
                self.joint_pub.publish(msg)
                self.rate.sleep()

                sq_error = np.sum((self.joint_pos - target_joint_pos) ** 2)
                rospy.loginfo(
                    f"[INFO] Moving robot... Squared joint error: {sq_error:.6f}"
                )
                if sq_error < self.joint_error_threshold:
                    break

            if i % 10 == 0:
                rospy.loginfo(
                    f"[INFO]: Executing step {(i/total_steps)*100:.1f}% Complete."
                )

        rospy.loginfo("[INFO]: Absolute Playback complete.")


if __name__ == "__main__":
    try:
        player = JetHexaAbsolutePlayer()
        player.initialize_robot_for_replay()
        player.play_trajectory()
        player.stop_robot()
    except rospy.ROSInterruptException:
        pass
