#!/usr/bin/env python3
"""
JET-HEXA ABSOLUTE TRAJECTORY PLAYBACK (SIMULTANEOUS 18-DOF MODE)
----------------------------
Functionality:
1. Finds the most recent .npz/.npy data file.
2. Extracts the 18-DOF absolute joint targets from the state vector.
3. Publishes all 18 joint targets simultaneously to the robot.
"""

import os
import sys
from pathlib import Path

# Resolve repo root and chdir there so "models/test_traj.npz" and the
# `from base import Base` import keep working after the move into scripts/.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

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
        rospy.init_node("jethexa_abs_player", anonymous=True)
        super().__init__(config=config)

    def play_trajectory(self):
        abs_joint_targets = self.test_data["states"][:, 6:24]
        total_steps = len(abs_joint_targets)

        # Execution Loop
        rospy.loginfo(f"[INFO]: Playing Trajectory with {total_steps} execution steps.")
        for i in range(total_steps):
            if rospy.is_shutdown():
                break

            target_joint_pos = abs_joint_targets[i]
            target_joint_pos = self.check_valid_joint_angle(
                target_joint_pos, terminate_on_invalid=False
            )
            target_joint_pos = target_joint_pos.tolist()

            # --- Move ALL 18 joints simultaneously ---
            msg = JointCommand()
            msg.target = target_joint_pos
            msg.duration = self.duration

            # Keep publishing and checking until the error is within the threshold
            j = 0
            while not rospy.is_shutdown():
                self.joint_abs_pub.publish(msg)
                self.rate.sleep()

                sq_error = np.sum((self.joint_pos - target_joint_pos) ** 2)
                if (
                    j % 5 == 0 and sq_error > 0
                ):  # Log every 5 iterations to avoid spamming
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

        rospy.loginfo("[INFO]: Absolute Playback complete.")


if __name__ == "__main__":
    try:
        player = JetHexaAbsolutePlayer()
        player.initialize_robot_for_replay()
        player.play_trajectory()
        player.stop_robot()
    except rospy.ROSInterruptException:
        pass
