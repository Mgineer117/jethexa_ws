#!/usr/bin/env python3
"""
Bring-up sanity check for the relative-joint command path.

Publishes small random JointCommand offsets on
/jethexa_controller/set_joints_relative, alternating between the two
tripod groups so the robot wiggles in place. Use it to confirm the
custom JointCommand interface is reachable from the PC and that the
robot is responding before running any of the real playback or control
scripts.
"""

import random

import rospy

from jethexa_controller_interfaces.msg import JointCommand


class JetHexaGroupNoiseInjector:
    def __init__(self):
        # Initialize node - anonymous=True allows multiple runs without naming conflicts
        rospy.init_node("jethexa_group_noise_injector", anonymous=True)

        self.gait_type = "tripod"  # For future expansion to other gaits

        if self.gait_type == "tripod":
            self.group_a = [0, 1, 2, 6, 7, 8, 12, 13, 14]
            self.group_b = [3, 4, 5, 9, 10, 11, 15, 16, 17]
        self.groups = [("Group A", self.group_a), ("Group B", self.group_b)]

        self.topic_name = "/jethexa_controller/set_joints_relative"
        self.joint_pub = rospy.Publisher(self.topic_name, JointCommand, queue_size=1)

        rospy.loginfo(f"Connecting to {self.topic_name}...")
        # Essential for wireless: Give the Master time to link the PC and Robot
        rospy.sleep(1.5)

    def send_group_command(self, joint_indices, noise_values, duration=0.5):
        """Builds and sends a relative JointCommand for a specific group of joints."""
        msg = JointCommand()

        # Create the 18-element vector of zeros
        target_array = [0.0] * 18

        # Map the noise values to their specific joint indices
        for idx, noise in zip(joint_indices, noise_values):
            if 0 <= idx < 18:
                target_array[idx] = float(noise)
            else:
                rospy.logerr(f"Invalid joint index: {idx}")

        msg.target = target_array
        msg.duration = float(duration)

        self.joint_pub.publish(msg)

    def run_noise_sequence(self, max_displacement=0.3, duration=0.5):
        """Injects noise into Group A, neutralizes, then Group B."""
        rospy.loginfo(
            f"Starting Noise Sequence. Max Amplitude: +/-{max_displacement} rad"
        )

        # Define the joint indices for the two Tripod groups
        # Group A: Leg 1 (0-2), Leg 3 (6-8), Leg 5 (12-14)

        # Group B: Leg 2 (3-5), Leg 4 (9-11), Leg 6 (15-17)

        for group_name, indices in self.groups:
            if rospy.is_shutdown():
                break

            rospy.loginfo(f"--- Injecting Noise into {group_name} ---")

            # 1. Generate random noise vector for this specific group
            noise_vector = [
                random.uniform(-max_displacement, max_displacement) for _ in indices
            ]

            # Print a sample of the noise to the terminal for debugging
            rospy.loginfo(
                f"Noise applied: {[round(n, 2) for n in noise_vector[:3]]}..."
            )

            # 2. Perturb the group
            self.send_group_command(indices, noise_vector, duration)
            rospy.sleep(duration + 0.1)

            # 3. Return to neutral (Exact negative of the noise applied)
            rospy.loginfo(f"Reversing noise to return {group_name} to neutral...")
            reverse_vector = [-n for n in noise_vector]
            self.send_group_command(indices, reverse_vector, duration)
            rospy.sleep(duration + 0.1)

        rospy.loginfo("Group noise sequence complete.")


if __name__ == "__main__":
    try:
        # 0.3 rad is about 17 degrees - noticeable but safe
        injector = JetHexaGroupNoiseInjector()
        injector.run_noise_sequence(max_displacement=0.3, duration=0.4)
    except rospy.ROSInterruptException:
        pass
