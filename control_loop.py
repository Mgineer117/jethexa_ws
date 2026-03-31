#!/usr/bin/env python3
import rospy
import torch
import torch.nn as nn
import numpy as np
from sensor_msgs.msg import JointState, Imu
from jethexa_controller_interfaces.msg import JointCommand
import time
import os
import argparse

class JetHexaRLCollector:
    def __init__(self, model_path, hz=25.0):
        rospy.init_node("jethexa_rl_collector", anonymous=True)

        self.output_dir = "hexapod_data"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # Timing and Rates
        self.hz = hz
        self.dt = 1.0 / self.hz
        self.rate = rospy.Rate(self.hz)

        # Data Buffers
        self.recorded_states = []
        self.recorded_actions = []

        # Robot State Variables
        self.joint_pos = np.zeros(18)
        self.prev_joint_pos = np.zeros(18)
        self.base_quat = np.array([0.0, 0.0, 0.0, 1.0])
        self.sensors_ready = False

        # Load PyTorch Model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = self.load_policy(model_path)
        
        # ROS Communication
        self.joint_pub = rospy.Publisher(
            "/jethexa_controller/set_joints_relative", 
            JointCommand, 
            queue_size=1, 
            tcp_nodelay=True
        )
        rospy.Subscriber("/joint_states", JointState, self.joint_cb)
        rospy.Subscriber("/imu/filtered", Imu, self.imu_cb)

        # --- WARMUP SEQUENCE ---
        self.warmup_controller()

    def load_policy(self, path):
        # NOTE: Replace 'YourModelClass' with your actual architecture import
        # model = YourModelClass().to(self.device)
        # model.load_state_dict(torch.load(path))
        # model.eval()
        rospy.loginfo(f"Loading policy from {path}...")
        # Placeholder for actual model loading
        return None 

    def warmup_controller(self):
        """Primes the IK engine by sending current-state heartbeats."""
        rospy.loginfo("Waiting for sensor stream...")
        while not self.sensors_ready and not rospy.is_shutdown():
            rospy.sleep(0.1)

        rospy.loginfo("Priming controller with 1.5s heartbeat...")
        msg = JointCommand()
        msg.target = [0.0] * 18 # 0 delta = stay where you are
        msg.duration = self.dt
        
        for _ in range(int(1.5 * self.hz)):
            self.joint_pub.publish(msg)
            rospy.sleep(self.dt)
        
        self.prev_joint_pos = np.copy(self.joint_pos)
        rospy.loginfo("Warmup complete. Controller is 'hot'.")

    def joint_cb(self, msg):
        self.joint_pos = np.array(msg.position)
        self.sensors_ready = True

    def imu_cb(self, msg):
        q = msg.orientation
        self.base_quat = np.array([q.x, q.y, q.z, q.w])

    def get_current_state(self):
        """Extracts 40-dim state vector."""
        joint_vel = (self.joint_pos - self.prev_joint_pos) / self.dt
        state = np.concatenate([self.base_quat, self.joint_pos, joint_vel])
        # We don't update prev_joint_pos here; we do it in the main loop
        return state

    def get_action_from_policy(self, state):
        """
        Policy outputs Joint Velocity (rad/s).
        We return relative delta (rad) = Velocity * dt
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # action_rate = self.policy(state_tensor).cpu().numpy().squeeze()
            action_rate = np.random.uniform(-0.5, 0.5, 18) # Mock inference
        
        # Scale by dt to get the relative displacement for this timestep
        return action_rate * self.dt

    def stop_robot(self):
        msg = JointCommand()
        msg.target = [0.0] * 18
        msg.duration = 0.5
        self.joint_pub.publish(msg)
        rospy.loginfo("Robot locked at current pose.")

    def run_rollout(self, duration=10.0):
        self.recorded_states = []
        self.recorded_actions = []
        start_time = rospy.get_time()

        rospy.loginfo("Starting Policy Rollout...")

        while not rospy.is_shutdown():
            elapsed = rospy.get_time() - start_time
            if elapsed >= duration:
                break

            # 1. Get current state and policy decision
            state = self.get_current_state()
            action_deltas = self.get_action_from_policy(state)

            # 2. Command the robot
            msg = JointCommand()
            msg.target = action_deltas.tolist()
            msg.duration = self.dt * 0.95 # Slightly faster execution to prevent lag
            self.joint_pub.publish(msg)

            # 3. Record (Store the rate/action provided by policy)
            self.recorded_states.append(state)
            self.recorded_actions.append(action_deltas / self.dt) # Store as rate

            # 4. Prepare for next iteration
            self.prev_joint_pos = np.copy(self.joint_pos)
            self.rate.sleep()

        self.stop_robot()
        self.save_data()

    def save_data(self):
        if not self.recorded_states: return
        ts = int(time.time())
        filename = os.path.join(self.output_dir, f"rl_policy_rollout_{ts}.npz")
        np.savez_compressed(
            filename,
            states=np.array(self.recorded_states),
            actions=np.array(self.recorded_actions)
        )
        rospy.loginfo(f"Data saved to {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="policy.pth")
    parser.add_argument("--duration", type=float, default=15.0)
    args = parser.parse_args()

    try:
        collector = JetHexaRLCollector(model_path=args.model, hz=25.0)
        collector.run_rollout(duration=args.duration)
    except rospy.ROSInterruptException:
        pass