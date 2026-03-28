import rospy
from jethexa_controller_interfaces.msg import JointCommand


class JetHexaRelativeDebugger:
    def __init__(self):
        # Initialize node - anonymous=True allows multiple runs without naming conflicts
        rospy.init_node("jethexa_relative_debugger", anonymous=True)

        # UPDATED TOPIC: Matches the output from your 'rostopic list'
        self.topic_name = "/jethexa_controller/set_joints_relative"

        self.joint_pub = rospy.Publisher(self.topic_name, JointCommand, queue_size=1)

        rospy.loginfo(f"Connecting to {self.topic_name}...")
        # Essential for wireless: Give the Master time to link the PC and Robot
        rospy.sleep(1.5)

    def send_relative_command(self, joint_idx, delta_value, duration=0.5):
        """Builds and sends a relative JointCommand for a single joint."""
        msg = JointCommand()

        # Create the 18-element vector required by the JetHexa controller
        target_array = [0.0] * 18

        # Guard against index errors
        if 0 <= joint_idx < 18:
            target_array[joint_idx] = float(delta_value)
        else:
            rospy.logerr(f"Invalid joint index: {joint_idx}")
            return

        msg.target = target_array
        msg.duration = float(duration)

        self.joint_pub.publish(msg)
        rospy.loginfo(f"Moving Joint [{joint_idx:02d}] by {delta_value:+.2f} rad")

    def run_debug_sequence(self, displacement=0.1, duration=0.5):
        """Iterates through all 18 joints, twitching them + then -."""
        rospy.loginfo(f"Starting Sequence. Amplitude: +/-{displacement} rad")

        for i in range(18):
            if rospy.is_shutdown():
                break

            # 1. Perturb positive
            self.send_relative_command(i, displacement, duration)
            rospy.sleep(duration + 0.1)

            # 2. Return to neutral (negative perturbation)
            self.send_relative_command(i, -displacement, duration)
            rospy.sleep(duration + 0.1)

        rospy.loginfo("Debug sequence complete.")


if __name__ == "__main__":
    try:
        # 0.1 rad is about 5.7 degrees - safe for testing
        debugger = JetHexaRelativeDebugger()
        debugger.run_debug_sequence(displacement=0.3, duration=0.4)
    except rospy.ROSInterruptException:
        pass
