#!/usr/bin/env python3
import argparse
import os
import time

# Import matplotlib and set to headless mode for ROS execution
import matplotlib
import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import Imu, JointState
from std_msgs.msg import String

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jethexa_controller_interfaces.msg import JointCommand
from parameters import HZ


class JetHexaDataCollector:
    def __init__(self, hz=10.0):
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
        self.base_pos = np.zeros(3)
        self.base_orientation = np.zeros(3)  # [Roll, Pitch, Yaw]
        self.joint_pos = np.zeros(18)
        self.prev_joint_pos = np.zeros(18)

        # Subscribers
        rospy.Subscriber("/joint_states", JointState, self.joint_cb)
        # rospy.Subscriber("/imu/filtered", Imu, self.imu_cb)
        rospy.Subscriber("/qualysis/jethexa", PoseStamped, self.base_pos_cb)

        # Publishers
        self.cmd_pub = rospy.Publisher(
            "/jethexa_controller/cmd_vel", Twist, queue_size=1
        )

        rospy.loginfo("Waiting for sensors...")
        rospy.sleep(2.0)
        self.prev_joint_pos = np.copy(self.joint_pos)

        # Waking up gait engine
        rospy.loginfo("Waking up gait engine...")
        warmup_cmd = Twist()
        for _ in range(3):
            self.cmd_pub.publish(warmup_cmd)
            rospy.sleep(1.0)
        rospy.loginfo("Gait engine ready.")

    def base_pos_cb(self, msg):
        self.base_pos = np.array(
            [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        )
        self.base_orientation = np.array(
            [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]
        )

    def joint_cb(self, msg):
        self.joint_pos = np.array(msg.position)[:18]

    def imu_cb(self, msg):
        """Converts incoming IMU quaternions to Euler angles for the 24D state space"""
        q = msg.orientation
        rot = R.fromquat([q.x, q.y, q.z, q.w])
        self.base_euler = rot.as_euler(
            "xyz", degrees=False
        )  # Returns Roll, Pitch, Yaw in radians

    def get_current_state(self):
        """
        24-dim State Vector:
        - [0:3]   : Position (x,y,z)
        - [3:6]   : Orientation (\phi, \theta, \psi)
        - [6:24]  : Joint Positions (rad)
        """
        # wrapped_diff = (self.joint_pos - self.prev_joint_pos + np.pi) % (
        #     2 * np.pi
        # ) - np.pi

        # 3. Calculate true velocity
        control = (self.joint_pos - self.prev_joint_pos) / self.dt

        state = np.concatenate([self.base_pos, self.base_orientation, self.joint_pos])
        self.prev_joint_pos = np.copy(self.joint_pos)
        return state, control

    def stop_robot(self):
        rospy.loginfo("Initiating hard stop and freeze sequence...")
        stop_twist = Twist()
        for _ in range(5):
            self.cmd_pub.publish(stop_twist)
            rospy.sleep(0.1)

    def _run_collection(self, mode, duration, get_cmd_func):
        self.recorded_states = []
        self.recorded_controls = []
        start_time = rospy.get_time()

        rospy.loginfo(f"Starting {mode} collection...")

        while not rospy.is_shutdown() and (rospy.get_time() - start_time) < duration:
            elapsed = rospy.get_time() - start_time

            cmd = get_cmd_func(elapsed)
            self.cmd_pub.publish(cmd)

            state, control = self.get_current_state()
            self.recorded_states.append(state)
            self.recorded_controls.append(control)

            self.rate.sleep()

            rospy.loginfo(
                f"Elapsed: {elapsed:.2f}s, Steps: {len(self.recorded_states)}"
            )

        self.stop_robot()
        self.save_data(mode)

    def collect_sinusoidal_turning(self, duration=30.0):
        s_vx, s_vy = np.random.uniform(0.02, 0.05), 0.0
        s_amp, s_freq = np.random.uniform(0.01, 0.05), 1 / 30

        def cmd_logic(t):
            s_wz = s_amp * np.sin(2 * np.pi * s_freq * t)
            c = Twist()
            c.linear.x, c.linear.y = s_vx, s_vy
            c.angular.z = s_wz
            return c

        self._run_collection("turning", duration, cmd_logic)

    def collect_smooth_acceleration(self, duration=30.0):
        max_vx, max_vy = np.random.uniform(0.02, 0.05), 0.0
        s_wz, s_freq = 0.0, 1 / 30

        def cmd_logic(t):
            ramp = (np.sin(2 * np.pi * s_freq * t) + 1.0) / 2.0
            c = Twist()
            c.linear.x, c.linear.y, c.angular.z = max_vx * ramp, max_vy * ramp, s_wz
            return c

        self._run_collection("accel", duration, cmd_logic)

    def collect_combined_stochastic(self, duration=30.0):
        max_vx, max_vy = np.random.uniform(0.02, 0.05), 0.0
        s_amp, s_freq = np.random.uniform(0.02, 0.05), 1 / 30

        def cmd_logic(t):
            xy_ramp = (np.sin(2 * np.pi * s_freq * t) + 1.0) / 2.0
            z_ramp = np.cos(2 * np.pi * s_freq * t)
            c = Twist()
            c.linear.x, c.linear.y, c.angular.z = (
                max_vx * xy_ramp,
                max_vy * xy_ramp,
                s_amp * z_ramp,
            )
            return c

        self._run_collection("combined", duration, cmd_logic)

    def plot_and_save(self, states_arr, save_path):
        """Generates an 8x3 grid plot of all 24 dimensions and saves it as a PNG."""
        fig, axs = plt.subplots(8, 3, figsize=(16, 20), sharex=True)
        fig.suptitle(
            f"Hexapod Trajectory: {os.path.basename(save_path)}", fontsize=18, y=0.98
        )

        axs_flat = axs.flatten()
        labels = [
            "Base Pos X",
            "Base Pos Y",
            "Base Pos Z",
            "Base Roll",
            "Base Pitch",
            "Base Yaw",
        ]
        labels += [f"Joint {i}" for i in range(1, 19)]

        timesteps = np.arange(states_arr.shape[0]) * self.dt

        for i in range(24):
            axs_flat[i].plot(timesteps, states_arr[:, i], linewidth=1.5, color="b")
            axs_flat[i].set_title(labels[i], fontsize=10)
            axs_flat[i].grid(True, alpha=0.3)

            if i < 3:
                axs_flat[i].set_ylabel("m", fontsize=8)
            else:
                axs_flat[i].set_ylabel("rad", fontsize=8)

        for i in range(21, 24):
            axs_flat[i].set_xlabel("Time (seconds)")

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(save_path)
        plt.close(fig)

        rospy.loginfo(f"Plot saved to {save_path}")

    def save_data(self, mode):
        if not self.recorded_states or not self.recorded_controls:
            rospy.logwarn("No data to save. States or Controls are empty.")
            return

        ts = int(time.time())
        base_filename = f"{mode}_{ts}"
        npz_filepath = os.path.join(self.output_dir, f"{base_filename}.npz")
        png_filepath = os.path.join(self.output_dir, f"{base_filename}.png")

        states_arr = np.array(self.recorded_states)
        controls_arr = np.array(self.recorded_controls)

        # 1. Save NPZ file
        np.savez(
            npz_filepath,
            states=states_arr,
            controls=controls_arr,
        )
        rospy.loginfo(
            f"Saved {mode} data -> States: {len(states_arr)}, Controls: {len(controls_arr)} steps."
        )

        # 2. Sanity Checks
        is_valid = True

        if np.isnan(states_arr).any():
            rospy.logerr(f"[SANITY CHECK FAILED] {npz_filepath} contains NaN values!")
            is_valid = False

        if states_arr.shape[0] < 300:
            rospy.logerr(
                f"[SANITY CHECK FAILED] {npz_filepath} length ({states_arr.shape[0]}) is less than 300!"
            )
            is_valid = False

        # 3. Plot if valid
        if is_valid:
            rospy.loginfo("Sanity checks passed. Generating trajectory plot...")
            self.plot_and_save(states_arr, png_filepath)
        else:
            rospy.logwarn(
                "Skipping trajectory plot generation due to failed sanity checks."
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", type=str, choices=["turning", "accel", "combined"], default="turning"
    )
    parser.add_argument("--duration", type=float, default=33.0)
    args = parser.parse_args()

    try:
        collector = JetHexaDataCollector(hz=HZ)
        if args.mode == "turning":
            collector.collect_sinusoidal_turning(duration=args.duration)
        elif args.mode == "accel":
            collector.collect_smooth_acceleration(duration=args.duration)
        elif args.mode == "combined":
            collector.collect_combined_stochastic(duration=args.duration)
    except rospy.ROSInterruptException:
        pass
