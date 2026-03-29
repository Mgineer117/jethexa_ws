#!/usr/bin/env python3
import rospy
import numpy as np
import time
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Twist


class JetHexaDiagnostic:
    def __init__(self):
        rospy.init_node("jethexa_diagnostic_tool", anonymous=True)

        # --- Network Stats Variables ---
        self.intervals = []
        self.last_sensor_time = None

        # --- Response Latency Variables ---
        self.cmd_pub = rospy.Publisher(
            "/jethexa_controller/cmd_vel", Twist, queue_size=1
        )
        self.initial_joints = None
        self.sent_time = None
        self.response_latencies = []
        self.movement_detected = False

        # Subscribers
        rospy.Subscriber("/imu/filtered", Imu, self.network_cb)
        rospy.Subscriber("/joint_states", JointState, self.joint_cb)

        rospy.loginfo("Diagnostic Tool Ready.")
        rospy.loginfo("1. Monitoring IMU network stability...")
        rospy.loginfo("2. Preparing Response Latency test pulses...")

    def network_cb(self, msg):
        """Calculates time between incoming sensor packets (Jitter)"""
        current_time = time.time()
        if self.last_sensor_time is not None:
            self.intervals.append(current_time - self.last_sensor_time)
        self.last_sensor_time = current_time

    def joint_cb(self, msg):
        """Monitors joints for movement after a command is sent"""
        if self.sent_time is not None and not self.movement_detected:
            current_pos = np.array(msg.position)
            # Threshold: 0.005 rad (~0.3 degrees) to filter out noise
            if np.any(np.abs(current_pos - self.initial_joints) > 0.005):
                latency = time.time() - self.sent_time
                self.response_latencies.append(latency)
                self.movement_detected = True

    def run_latency_test(self, num_tests=5):
        """Sends sharp movement pulses to measure physical response delay"""
        for i in range(num_tests):
            if rospy.is_shutdown():
                break

            rospy.loginfo(f"Test Pulse {i+1}/{num_tests}...")
            # Capture baseline
            self.initial_joints = np.array(
                rospy.wait_for_message("/joint_states", JointState).position
            )

            self.movement_detected = False
            self.sent_time = time.time()

            # Send a quick twist command
            pulse = Twist()
            pulse.angular.z = 0.8
            self.cmd_pub.publish(pulse)

            # Wait for detection or 1s timeout
            timeout = time.time() + 1.0
            while not self.movement_detected and time.time() < timeout:
                rospy.sleep(0.01)

            # Stop robot and settle
            self.cmd_pub.publish(Twist())
            rospy.sleep(1.5)

    def report(self):
        print("\n" + "=" * 50)
        print("          JET-HEXA SYSTEM DIAGNOSTICS")
        print("=" * 50)

        # --- Network Analysis ---
        if self.intervals:
            ints = np.array(self.intervals)
            avg_hz = 1.0 / np.mean(ints)
            max_gap = np.max(ints)
            lower_bound = 1.0 / max_gap

            print(f"[NETWORK: /imu/filtered]")
            print(f"  Avg Frequency: {avg_hz:.2f} Hz")
            print(f"  Worst Case:    {lower_bound:.2f} Hz (Gap: {max_gap:.4f}s)")
            status = "PASS" if max_gap < 0.1 else "FAIL (Jitter too high)"
            print(f"  10Hz Stability: {status}")
        else:
            print("[NETWORK] No sensor data received.")

        print("-" * 50)

        # --- Response Analysis ---
        if self.response_latencies:
            lats = np.array(self.response_latencies) * 1000  # convert to ms
            print(f"[RESPONSE: cmd_vel -> joint_states]")
            print(f"  Avg Latency:  {np.mean(lats):.2f} ms")
            print(f"  Max Latency:  {np.max(lats):.2f} ms")

            # Control Logic Check
            if np.mean(lats) > 100:
                print("  STATUS: HIGH LATENCY. Policy is 'lagging'.")
            else:
                print("  STATUS: GOOD. Response is within one 10Hz frame.")
        else:
            print("[RESPONSE] No movement pulses detected.")

        print("=" * 50 + "\n")


if __name__ == "__main__":
    diag = JetHexaDiagnostic()
    try:
        # Run the pulses for a few seconds while network logs in background
        diag.run_latency_test(num_tests=5)
    except KeyboardInterrupt:
        pass
    finally:
        diag.report()
