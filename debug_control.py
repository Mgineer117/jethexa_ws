#!/usr/bin/env python3
import rospy
from jethexa_controller_interfaces.msg import JointCommand


class JetHexaRelativeDebugger:
    def __init__(self):
        rospy.init_node("jethexa_relative_debugger", anonymous=True)

        # Publisher for RELATIVE joint commands
        self.joint_pub = rospy.Publisher(
            "/jethexa_controller/joint_relative", JointCommand, queue_size=1
        )

        rospy.loginfo("Waiting for joint_relative publisher to connect...")
        rospy.sleep(2.0)  # Give ROS time to register the publisher

    def send_relative_command(self, joint_idx, delta_value, duration=0.5):
        """Builds and sends a relative JointCommand for a single joint."""
        msg = JointCommand()

        # Initialize an array of 18 zeros.
        # In relative mode, 0.0 means "don't move this joint".
        target_array = [0.0] * 18

        # Set the specific joint to the delta value (+0.1 or -0.1)
        target_array[joint_idx] = delta_value

        msg.target = target_array
        msg.duration = duration

        self.joint_pub.publish(msg)
        rospy.loginfo(f"  Joint [{joint_idx}] moving by {delta_value:+.2f} rad")

    def run_debug_sequence(self, displacement=0.1, duration=0.5):
        """Iterates through all 18 joints, twitching them + then -."""
        rospy.loginfo(
            f"Starting Relative Debug Sequence. Movement: +/-{displacement} rad"
        )

        for i in range(18):
            if rospy.is_shutdown():
                rospy.logwarn("Sequence interrupted by user.")
                break

            rospy.loginfo(f"--- Testing Joint [{i}] ---")

            # 1. Move by +displacement
            self.send_relative_command(i, displacement, duration)
            rospy.sleep(duration + 0.2)  # Sleep slightly longer to ensure completion

            # 2. Move by -displacement (reversing the previous move to return to start)
            self.send_relative_command(i, -displacement, duration)
            rospy.sleep(duration + 0.2)

            rospy.sleep(0.1)  # Brief pause before moving to the next joint

        rospy.loginfo(
            "Debug sequence complete. All joints returned to starting positions."
        )


if __name__ == "__main__":
    try:
        debugger = JetHexaRelativeDebugger()
        # Using 0.1 rad (~5.7 degrees) and 0.5s duration
        debugger.run_debug_sequence(displacement=0.1, duration=0.5)
    except rospy.ROSInterruptException:
        pass
