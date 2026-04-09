import glob
import os

import numpy as np
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Imu, JointState

from jethexa_controller_interfaces.msg import JointCommand

INIT_JOINT_POS = [
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


HZ = 10

GROUP_A = [0, 1, 2, 6, 7, 8, 12, 13, 14]
GROUP_B = [3, 4, 5, 9, 10, 11, 15, 16, 17]


class Base:
    def __init__(self, config: dict):

        # Configurations
        self.use_mocap = config.get("use_mocap", False)
        self.use_imu = config.get("use_imu", False)

        self.listen_joint_states = config.get("listen_joint_states", False)

        self.use_abs_joint_commands = config.get("use_abs_joint_commands", False)
        self.use_rel_joint_commands = config.get("use_rel_joint_commands", False)

        self.load_test_traj = config.get("load_test_traj", False)
        self.load_recent_traj = config.get("load_recent_traj", False)

        # Initialise readiness flags unconditionally so check_sensors() and
        # any other method can always read them safely.
        self.mocap_ready = False
        self.imu_ready = False
        self.joints_ready = False

        # ROS Subscribers and Publishers
        if self.use_mocap:
            rospy.Subscriber("/qualysis/jethexa", PoseStamped, self.base_pos_cb)
            self.base_pos = np.zeros(2)
            self.base_attitude = np.zeros(3)

        if self.use_imu:
            rospy.Subscriber("/imu/filtered", Imu, self.imu_cb)
            self.base_quat = np.zeros(4)  # FIX: was only assigned in imu_cb

        if self.listen_joint_states:
            rospy.Subscriber("/joint_states", JointState, self.joint_cb)
            self.joint_pos = np.copy(INIT_JOINT_POS)  # safe default until first msg

        # Exactly one joint-command publisher is exposed as self.joint_pub so
        # that the rest of the class doesn't need to know which topic is live.
        # Absolute commands take priority when both flags are set.
        if self.use_abs_joint_commands:
            self.joint_abs_pub = rospy.Publisher(
                "/jethexa_controller/set_joints_raw",
                JointCommand,
                queue_size=1,
                tcp_nodelay=True,
            )
            self.joint_pub = self.joint_abs_pub  # FIX: was never defined

        if self.use_rel_joint_commands:
            self.joint_rel_pub = rospy.Publisher(
                "/jethexa_controller/set_joints_relative",
                JointCommand,
                queue_size=1,
                tcp_nodelay=True,
            )
            # Only fall back to relative publisher when no absolute publisher exists.
            if not self.use_abs_joint_commands:
                self.joint_pub = self.joint_rel_pub  # FIX: was never defined

        if not self.use_abs_joint_commands and not self.use_rel_joint_commands:
            # Neither publisher requested – create a harmless sentinel so that
            # stop_robot() / check_sensors() do not crash if called anyway.
            self.joint_pub = None

        if self.load_test_traj:
            self.test_data = self.load_test_data()
        if self.load_recent_traj:
            self.recent_data = self.load_latest_data()

        # Parameters
        self.hz = HZ
        self.dt = 1.0 / self.hz
        self.duration = self.dt * 0.95
        self.rate = rospy.Rate(self.hz)

        self.init_joint_pos = INIT_JOINT_POS
        self.joint_error_threshold = 1e-3

        self.check_sensors()

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def base_pos_cb(self, msg):
        self.base_pos = np.array([msg.pose.position.x, msg.pose.position.y])
        self.base_attitude = np.array(
            [msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z]
        )
        self.mocap_ready = True

    def imu_cb(self, msg):
        """Stores incoming IMU quaternion for the state vector."""
        q = msg.orientation
        self.base_quat = np.array([q.x, q.y, q.z, q.w])  # FIX: was base_quad (typo)
        self.imu_ready = True

    def joint_cb(self, msg):
        self.joint_pos = np.array(msg.position)[:18]
        self.joints_ready = True

    # ------------------------------------------------------------------
    # Data loading helpers
    # ------------------------------------------------------------------

    def load_latest_data(self, filename=None):
        """
        24-dim State Vector:
        - [0:3]   : Position (x, y, z)
        - [3:6]   : Orientation (theta, phi, psi)
        - [6:24]  : Joint Positions (18 DOF)
        """
        if filename:
            return np.load(os.path.join("hexapod_data", filename))

        files = glob.glob(os.path.join("models", "*.npz"))
        if not files:
            rospy.logerr(f"No .npz files found in: hexapod_data/ ")
            return None
        latest_file = max(files, key=os.path.getmtime)
        return np.load(latest_file)

    def load_test_data(self):
        """
        24-dim State Vector:
        - [0:3]   : Position (x, y, z)
        - [3:6]   : Orientation (theta, phi, psi)
        - [6:24]  : Joint Positions (18 DOF)
        """
        test_file = os.path.join("models/test_traj.npz")
        if not os.path.exists(test_file):
            rospy.logerr(f"Test file not found: {test_file}")
            return None
        return np.load(test_file)

    # ------------------------------------------------------------------
    # Sensor readiness
    # ------------------------------------------------------------------

    def check_sensors(self):
        """
        Block until every *enabled* sensor has delivered at least one message,
        then prime the joint controller with a brief warmup sequence.

        Readiness conditions driven by config flags
        ─────────────────────────────────────────────
        listen_joint_states → joints_ready
        use_mocap           → mocap_ready
        use_imu             → imu_ready
        """
        rospy.loginfo("Waiting for sensor streams...")

        while not rospy.is_shutdown():
            # Build the readiness predicate from only the *active* sensors.
            # FIX: original code always required joints_ready AND mocap_ready
            #      regardless of which sensors were actually enabled.
            joint_ok = self.joints_ready if self.listen_joint_states else True
            mocap_ok = self.mocap_ready if self.use_mocap else True
            imu_ok = self.imu_ready if self.use_imu else True

            if joint_ok and mocap_ok and imu_ok:
                break

            rospy.loginfo(
                f"  - Joints ready : {self.joints_ready} (required: {self.listen_joint_states})\n"
                f"  - Mocap ready  : {self.mocap_ready}  (required: {self.use_mocap})\n"
                f"  - IMU ready    : {self.imu_ready}    (required: {self.use_imu})"
            )
            rospy.sleep(0.1)

        rospy.loginfo("All required sensors ready.")

    # ------------------------------------------------------------------
    # Control helpers
    # ------------------------------------------------------------------

    def initialize_robot_for_replay(self):
        """Moves the robot to the initial pose of the trajectory for a smooth start."""
        if self.joint_pub is None:
            rospy.logwarn(
                "initialize_robot_for_replay() called but no joint publisher is configured."
            )
            return

        # retrieve first frame joint angle
        if self.load_recent_traj and self.load_test_traj:
            rospy.logwarn(
                "Both recent and test trajectories are loaded. Defaulting to test trajectory for initialization."
            )
            init_joint_pos = self.test_data["states"][0, 6:24]
        elif self.load_recent_traj:
            init_joint_pos = self.recent_data["states"][0, 6:24]
        elif self.load_test_traj:
            init_joint_pos = self.test_data["states"][0, 6:24]
        else:
            rospy.logwarn(
                "No trajectory data loaded. Using default INIT_JOINT_POS for initialization."
            )
            init_joint_pos = self.init_joint_pos

        target_joint_pos = init_joint_pos.tolist()

        msg = JointCommand()
        msg.target = target_joint_pos
        msg.duration = self.duration

        if self.listen_joint_states:
            # Keep publishing and checking until the error is within the threshold
            while not rospy.is_shutdown():
                self.joint_pub.publish(msg)
                self.rate.sleep()  # Sleep to maintain the loop rate

                sq_error = np.sum((self.joint_pos - target_joint_pos) ** 2)
                rospy.loginfo(
                    f"[INFO] Initializing robot... Squared joint error: {sq_error:.6f}"
                )
                if sq_error < self.joint_error_threshold:
                    break
        else:
            # Open-loop fallback: Publish repeatedly for 1 second to ensure delivery.
            # (A single publish in ROS often gets dropped).
            rospy.loginfo("[INFO] Initializing robot (Open-loop)...")
            for _ in range(int(self.hz)):
                if rospy.is_shutdown():
                    break
                self.joint_pub.publish(msg)
                self.rate.sleep()

    def stop_robot(self):
        """Send the robot back to its default standing pose."""
        if self.joint_pub is None:
            rospy.logwarn("stop_robot() called but no joint publisher is configured.")
            return

        target_joint_pos = self.init_joint_pos
        msg = JointCommand()
        msg.target = target_joint_pos
        msg.duration = self.duration

        if self.listen_joint_states:
            # Keep publishing and checking until the error is within the threshold
            while not rospy.is_shutdown():
                self.joint_pub.publish(msg)
                self.rate.sleep()  # Sleep to maintain the loop rate

                sq_error = np.sum((self.joint_pos - target_joint_pos) ** 2)
                rospy.loginfo(
                    f"[INFO] Stopping robot... Squared joint error: {sq_error:.6f}"
                )
                if sq_error < self.joint_error_threshold:
                    break
        else:
            # Open-loop fallback: Publish repeatedly for 1 second to ensure delivery.
            # (A single publish in ROS often gets dropped).
            rospy.loginfo("[INFO] Stopping robot (Open-loop)...")
            for _ in range(int(self.hz)):
                if rospy.is_shutdown():
                    break
                self.joint_pub.publish(msg)
                self.rate.sleep()
