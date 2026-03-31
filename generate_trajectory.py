#!/usr/bin/env python3
import argparse
import os
import time

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import String

from jethexa_controller_interfaces.msg import JointCommand


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
        self.recorded_controls = []

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
        self.joint_pub = rospy.Publisher(
            "/jethexa_controller/set_joints_raw",
            JointCommand,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.action_pub = rospy.Publisher(
            "/jethexa_controller/run_actionset", String, queue_size=1
        )

        self.init_joint_pos = [
            0.17120142291266816,
            0.7383813588273885,
            -0.574647577430417,
            0.0,
            0.7435823754497041,
            -0.5952505177281151,
            -0.17120142291266816,
            0.7383813588273885,
            -0.574647577430417,
            0.17120142291266838,
            0.7383813588273884,
            -0.5746475774304161,
            0.0,
            0.7435823754497041,
            -0.5952505177281151,
            -0.17120142291266838,
            0.7383813588273884,
            -0.5746475774304161,
        ]

        rospy.loginfo("Waiting for sensors...")
        rospy.sleep(2.0)
        self.prev_joint_pos = np.copy(self.joint_pos)

        # Add this to the end of your __init__
        rospy.loginfo("Waking up gait engine...")
        warmup_cmd = Twist()
        for _ in range(3):
            self.cmd_pub.publish(warmup_cmd)
            rospy.sleep(1.0)
        rospy.loginfo("Gait engine ready.")

    def joint_cb(self, msg):
        self.joint_pos = np.array(msg.position)[:18]

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
        control = self.joint_pos - self.prev_joint_pos
        joint_vel = control / self.dt
        state = np.concatenate([self.base_quat, self.joint_pos, joint_vel])

        # Update prev_pos AFTER calculating state for this step
        self.prev_joint_pos = np.copy(self.joint_pos)
        return state, control

    def stop_robot(self):
        rospy.loginfo("Initiating hard stop and freeze sequence...")

        # 1. Kill velocity (Spam it so the controller catches it)
        stop_twist = Twist()
        for _ in range(5):
            self.cmd_pub.publish(stop_twist)
            rospy.sleep(0.1)  # CRITICAL: Give the controller time to receive

        # 2. Kill the Gait Engine / IMU Balance loop
        # Sending this action forces the JetHexa out of the walking state machine
        # so it stops fighting your raw joint commands.
        reset_msg = String()
        reset_msg.data = "initial_pose"
        self.action_pub.publish(reset_msg)
        rospy.sleep(1.0)  # Wait for the legs to lock

        # 3. Publish your exact desired joint positions
        rospy.loginfo("Applying specific neutral joint coordinates...")
        msg_a = JointCommand()
        msg_a.target = self.init_joint_pos
        msg_a.duration = 1.0

        for _ in range(3):
            if rospy.is_shutdown():
                break
            self.joint_pub.publish(msg_a)
            rospy.sleep(
                0.5
            )  # CRITICAL: Sleep inside the loop so messages aren't dropped!

        rospy.loginfo("Robot successfully stopped and locked.")

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
            state, control = self.get_current_state()
            self.recorded_states.append(state)
            self.recorded_controls.append(control)

            self.rate.sleep()

        self.stop_robot()
        self.save_data(mode)

    def collect_sinusoidal_turning(self, duration=15.0):
        s_vx, s_vy = np.random.uniform(0.05, 0.1), 0.0
        s_amp, s_freq = np.random.uniform(0.1, 0.3), 1 / 30

        def cmd_logic(t):
            c = Twist()
            c.linear.x, c.linear.y = s_vx, s_vy
            c.angular.z = s_amp * np.cos(2 * np.pi * s_freq * t)
            rospy.loginfo(f"Time: {t:.2f}, omega_z: {np.cos(2 * np.pi * s_freq * t)}")
            return c

        self._run_collection("turning", duration, cmd_logic)

    def collect_smooth_acceleration(self, duration=15.0):
        max_vx, max_vy = 0.1, 0.0
        s_freq = np.random.uniform(0.1, 0.3)
        s_wz = 0.0

        def cmd_logic(t):
            ramp = (np.sin(s_freq * t) + 1.0) / 2.0
            c = Twist()
            c.linear.x, c.linear.y, c.angular.z = max_vx * ramp, max_vy * ramp, s_wz
            return c

        self._run_collection("accel", duration, cmd_logic)

    def collect_combined_stochastic(self, duration=15.0):
        max_vx, max_vy = 0.1, 0.0
        s_amp, s_freq = np.random.uniform(0.1, 0.3), np.random.uniform(0.1, 0.3)

        def cmd_logic(t):
            xy_ramp = (np.sin(s_freq * t) + 1.0) / 2.0
            z_ramp = np.sin(s_freq * t)
            c = Twist()
            c.linear.x, c.linear.y, c.angular.z = (
                max_vx * xy_ramp,
                max_vy * xy_ramp,
                s_amp * z_ramp,
            )
            return c

        self._run_collection("accel", duration, cmd_logic)

    def save_data(self, mode):
        # Check if either list is empty
        if not self.recorded_states or not self.recorded_controls:
            rospy.logwarn("No data to save. States or Controls are empty.")
            return

        ts = int(time.time())
        # Note the change from .npy to .npz for multi-array archives
        filename = os.path.join(self.output_dir, f"{mode}_{ts}.npz")

        # Save both lists as named arrays inside the .npz file
        np.savez(
            filename,
            states=np.array(self.recorded_states),
            controls=np.array(self.recorded_controls),
        )

        rospy.loginfo(
            f"Saved {mode} data -> States: {len(self.recorded_states)}, Controls: {len(self.recorded_controls)} steps."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, choices=["turning", "accel", "combined"], default="turning"
    )
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--hz", type=float, default=10.0)
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
