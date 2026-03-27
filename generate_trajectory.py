#!/usr/bin/env python3
import rospy
import numpy as np
from sensor_msgs.msg import JointState, Imu
from geometry_msgs.msg import Twist
from std_msgs.msg import String
import time
import os
import argparse


class JetHexaDataCollector:
    def __init__(self, hz=25.0):
        rospy.init_node("jethexa_data_collector", anonymous=True)

        self.output_dir = "hexapod_data"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Timing
        self.hz = hz
        self.dt = 1.0 / self.hz
        self.rate = rospy.Rate(self.hz)

        # Buffers for NPZ saving
        self.recorded_states = []

        # Sensor State
        self.joint_pos = np.zeros(18)
        self.prev_joint_pos = np.zeros(18)
        self.base_quat = np.array([0.0, 0.0, 0.0, 1.0])

        # Subscribers
        rospy.Subscriber("/joint_states", JointState, self.joint_cb)
        rospy.Subscriber("/imu/filtered", Imu, self.imu_cb)

        # Publishers
        self.cmd_pub = rospy.Publisher(
            "/jethexa_controller/cmd_vel", Twist, queue_size=1
        )
        self.action_pub = rospy.Publisher(
            "/jethexa_controller/run_actionset", String, queue_size=1
        )

        rospy.loginfo("Waiting for sensors...")
        rospy.sleep(2.0)
        self.prev_joint_pos = np.copy(self.joint_pos)

    def joint_cb(self, msg):
        self.joint_pos = np.array(msg.position)

    def imu_cb(self, msg):
        q = msg.orientation
        self.base_quat = np.array([q.x, q.y, q.z, q.w])

    def get_current_state(self):
        """
        40-dim State Vector:
        - [0:4]   : Quat (x,y,z,w)
        - [4:22]  : Joint Positions (rad)
        - [22:40] : Joint Velocities (rad/s)
        """
        joint_vel = (self.joint_pos - self.prev_joint_pos) / self.dt
        state = np.concatenate([self.base_quat, self.joint_pos, joint_vel])

        # Update prev_pos AFTER calculating state for this step
        self.prev_joint_pos = np.copy(self.joint_pos)
        return state

    def stop_robot(self):
        stop_twist = Twist()
        for _ in range(5):
            self.cmd_pub.publish(stop_twist)
            rospy.sleep(0.05)
        self.action_pub.publish("initial_pose")
        rospy.sleep(1.0)

    # --- Collection Methods (Simplified to call a single record loop) ---

    def _run_collection(self, mode, duration, get_cmd_func):
        self.recorded_states = []
        start_time = rospy.get_time()

        rospy.loginfo(f"Starting {mode} collection...")

        while not rospy.is_shutdown() and (rospy.get_time() - start_time) < duration:
            elapsed = rospy.get_time() - start_time

            # 1. Get the command from the specific mode logic
            cmd = get_cmd_func(elapsed)
            self.cmd_pub.publish(cmd)

            # 2. Record the current state (Observations)
            # Note: We aren't recording 'actions' here because cmd_vel
            # doesn't expose the Cartesian foot deltas we need for set_leg_relatively.
            self.recorded_states.append(self.get_current_state())

            self.rate.sleep()

        self.stop_robot()
        self.save_data(mode)

    def collect_sinusoidal_turning(self, duration=15.0):
        s_vx, s_vy = np.random.uniform(0.05, 0.1), np.random.uniform(-0.1, 0.1)
        s_amp, s_freq = np.random.uniform(0.1, 0.5), np.random.uniform(0.1, 1.0)

        def cmd_logic(t):
            c = Twist()
            c.linear.x, c.linear.y = s_vx, s_vy
            c.angular.z = s_amp * np.sin(s_freq * t)
            return c

        self._run_collection("turning", duration, cmd_logic)

    def collect_smooth_acceleration(self, duration=15.0):
        s_wz = np.random.uniform(-0.3, 0.3)
        max_vx, max_vy = np.random.uniform(0.05, 0.10), np.random.uniform(0.02, 0.04)
        s_freq = np.random.uniform(0.3, 1.0)

        def cmd_logic(t):
            ramp = (np.sin(s_freq * t) + 1.0) / 2.0
            c = Twist()
            c.linear.x, c.linear.y, c.angular.z = max_vx * ramp, max_vy * ramp, s_wz
            return c

        self._run_collection("accel", duration, cmd_logic)

    def collect_combined_stochastic(self, duration=20.0):
        max_vx, max_vy, max_wz = (
            np.random.uniform(0.04, 0.09),
            np.random.uniform(0.02, 0.04),
            np.random.uniform(0.3, 0.7),
        )
        f_linear, f_angular = np.random.uniform(0.4, 0.8), np.random.uniform(0.8, 1.5)

        def cmd_logic(t):
            l_ramp = (np.sin(f_linear * t) + 1.0) / 2.0
            c = Twist()
            c.linear.x, c.linear.y = max_vx * l_ramp, max_vy * l_ramp
            c.angular.z = max_wz * np.sin(f_angular * t)
            return c

        self._run_collection("combined", duration, cmd_logic)

    def save_data(self, mode):
        if not self.recorded_states:
            return
        ts = int(time.time())
        filename = os.path.join(self.output_dir, f"{mode}_{ts}.npy")
        np.save(filename, np.array(self.recorded_states))
        rospy.loginfo(f"Saved {mode} state data: {len(self.recorded_states)} steps.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, choices=["turning", "accel", "combined"], default="turning"
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--hz", type=float, default=25.0)
    args = parser.parse_args()

    try:
        collector = JetHexaDataCollector(hz=args.hz)
        if args.mode == "turning":
            collector.collect_sinusoidal_turning(duration=args.duration)
        elif args.mode == "accel":
            collector.collect_smooth_acceleration(duration=args.duration)
        elif args.mode == "combined":
            collector.collect_combined_stochastic(duration=args.duration)
    except rospy.ROSInterruptException:
        pass
