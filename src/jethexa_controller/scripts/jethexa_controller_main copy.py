#!/usr/bin/env python3
# coding: utf-8

import time
import rospy
import nav_msgs.msg as nav_msgs
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import (
    Quaternion,
    Point,
    Vector3,
    TransformStamped,
    TwistWithCovarianceStamped,
)

from jethexa_controller_interfaces import msg as jetmsg
from jethexa_controller_interfaces.srv import (
    SetPose1,
    SetPose1Request,
    SetPose1Response,
    SetPose2,
    SetPose2Request,
    SetPose2Response,
    PoseTransform,
    PoseTransformRequest,
    PoseTransformResponse,
)

import jethexa_sdk.buzzer as buzzer
from jethexa_controller import jethexa, build_in_pose, config
from jethexa_controller.z_voltage_publisher import VoltagePublisher
from jethexa_controller.z_joint_states_publisher import JointStatesPublisher
import geometry_msgs.msg


class jethexaControlNode:
    def __init__(self, node_name):
        rospy.init_node(node_name, anonymous=True)

        self.tf_prefix = rospy.get_param("~tf_prefix", "")
        self.tf_prefix = (self.tf_prefix + "/") if self.tf_prefix != "" else ""

        self.controller = jethexa.JetHexa(self)

        # Safety limits for servos in radians
        self.JOINT_MIN = -1.25
        self.JOINT_MAX = 1.25

        # SDK Joint Keys for Relative Lookup mapped to URDF names
        self.JOINT_MAP = [
            "coxa_joint_LF",
            "femur_joint_LF",
            "tibia_joint_LF",  # Left Front
            "coxa_joint_LM",
            "femur_joint_LM",
            "tibia_joint_LM",  # Left Middle
            "coxa_joint_LR",
            "femur_joint_LR",
            "tibia_joint_LR",  # Left Rear
            "coxa_joint_RR",
            "femur_joint_RR",
            "tibia_joint_RR",  # Right Rear
            "coxa_joint_RM",
            "femur_joint_RM",
            "tibia_joint_RM",  # Right Middle
            "coxa_joint_RF",
            "femur_joint_RF",
            "tibia_joint_RF",  # Right Front
        ]

        # Status publishers
        self.voltage_publisher = VoltagePublisher(node=self, rate=1)
        self.joint_states_publisher = JointStatesPublisher(node=self, rate=20)

        # Joint Position Control (Restored to custom JointCommand)
        self.set_joints_raw_sub = rospy.Subscriber(
            "/jethexa_controller/set_joints_raw",
            jetmsg.JointCommand,
            self.joint_absolute_cb,
        )
        self.set_joints_rel_sub = rospy.Subscriber(
            "/jethexa_controller/set_joints_relative",
            jetmsg.JointCommand,
            self.joint_relative_cb,
        )

        # Posture and Leg Control
        self.set_transform2_sub = rospy.Subscriber(
            "/jethexa_controller/pose_transform_euler",
            jetmsg.TransformEuler,
            self.pose_transform_euler_cb,
        )
        self.set_pose_euler_sub = rospy.Subscriber(
            "/jethexa_controller/set_pose_euler", jetmsg.Pose, self.set_pose_euler_cb
        )
        self.set_leg_position_sub = rospy.Subscriber(
            "/jethexa_controller/set_leg_absolute",
            jetmsg.LegPosition,
            self.set_leg_absolute_cb,
        )
        self.set_leg_position_re_sub = rospy.Subscriber(
            "/jethexa_controller/set_leg_relatively",
            jetmsg.LegPosition,
            self.set_leg_relatively_cb,
        )

        # Head Control
        self.set_head_absolute = rospy.Subscriber(
            "/jethexa_controller/set_head_absolute",
            jetmsg.TransformEuler,
            self.head_absolute_cb,
        )
        self.set_head_relatively = rospy.Subscriber(
            "/jethexa_controller/set_head_relatively",
            jetmsg.TransformEuler,
            self.head_relatively_cb,
        )

        # Movement and Actions
        self.traveling = rospy.Subscriber(
            "/jethexa_controller/traveling", jetmsg.Traveling, self.set_traveling_cb
        )
        self.run_action_sub = rospy.Subscriber(
            "/jethexa_controller/run_actionset",
            jetmsg.RunActionSet,
            self.run_action_set_sub_cb,
        )
        self.cmd_vel_sub = rospy.Subscriber(
            "/jethexa_controller/cmd_vel",
            geometry_msgs.msg.Twist,
            self.controller.cmd_vel,
        )

        # Services
        self.set_pose1_srv = rospy.Service("~set_pose_1", SetPose1, self.set_pose1_cb)
        self.set_pose2_srv = rospy.Service("~set_pose_2", SetPose2, self.set_pose2_cb)

        # Odometry
        self.odom_sub = rospy.Subscriber(
            "odom/filtered", nav_msgs.Odometry, self.odom_callback
        )

        rospy.Timer(rospy.Duration(2), self.on_start, oneshot=True)

    def on_start(self, _):
        self.controller.set_build_in_pose("DEFAULT_POSE", 2)
        buzzer.on()
        time.sleep(0.1)
        buzzer.off()

    def odom_callback(self, msg: nav_msgs.Odometry):
        o = msg.pose.pose.orientation
        r = R.from_quat((o.x, o.y, o.z, o.w))
        yaw = r.as_euler("xyz", degrees=False)
        self.controller.real_pose_yaw = yaw[-1]

    def joint_absolute_cb(self, msg: jetmsg.JointCommand):
        """Moves 18 joints to absolute positions in target[18]."""
        if len(msg.target) != 18:
            rospy.logwarn(
                f"Absolute command rejected: Expected 18 joints, got {len(msg.target)}"
            )
            return
        try:
            for i, target_angle in enumerate(msg.target):
                # Retrieve current state from SDK joints_state dict
                current_angle = self.controller.joints_state.get(self.JOINT_MAP[i], 0.0)

                # Smart Filter: Do not send hardware commands if we are already at the target
                if abs(target_angle - current_angle) < 1e-4:
                    continue

                # Clamp for hardware safety
                safe_angle = max(self.JOINT_MIN, min(self.JOINT_MAX, target_angle))
                # Servos are 1-indexed in SDK
                self.controller.set_joint(i + 1, safe_angle, msg.duration)
        except Exception as e:
            rospy.logerr(f"Joint Absolute Move Failed: {e}")

    def joint_relative_cb(self, msg: jetmsg.JointCommand):
        """Adds target[18] offsets to current joint positions."""
        if len(msg.target) != 18:
            rospy.logwarn(
                f"Relative command rejected: Expected 18 joints, got {len(msg.target)}"
            )
            return
        try:
            for i, offset in enumerate(msg.target):
                # Smart Filter: Skip the calculation and hardware command if offset is zero
                if abs(offset) < 1e-4:
                    continue

                # Retrieve current state from SDK joints_state dict
                current_angle = self.controller.joints_state.get(self.JOINT_MAP[i], 0.0)
                new_angle = current_angle + offset

                # Clamp and send
                safe_angle = max(self.JOINT_MIN, min(self.JOINT_MAX, new_angle))
                self.controller.set_joint(i + 1, safe_angle, msg.duration)
        except Exception as e:
            rospy.logerr(f"Joint Relative Move Failed: {e}")

    def head_absolute_cb(self, msg: jetmsg.TransformEuler):
        try:
            safe_yaw = max(-1.5, min(1.5, msg.rotation.z))
            safe_pitch = max(-0.35, min(0.6, msg.rotation.y))
            self.controller.set_joint(19, safe_yaw, msg.duration)
            self.controller.set_joint(20, safe_pitch, msg.duration)
        except Exception as e:
            rospy.logerr(e)

    def head_relatively_cb(self, msg: jetmsg.TransformEuler):
        try:
            new_yaw = self.controller.joints_state["head_pan_joint"] + msg.rotation.z
            new_pitch = self.controller.joints_state["head_tilt_joint"] + msg.rotation.y
            if new_pitch >= -0.35:
                self.controller.set_joint(19, new_yaw, msg.duration)
                self.controller.set_joint(20, new_pitch, msg.duration)
        except Exception as e:
            rospy.logerr(e)

    def set_traveling_cb(self, msg: jetmsg.Traveling):
        try:
            if msg.gait > 0:
                self.controller.set_step_mode(
                    msg.gait,
                    msg.stride,
                    msg.height,
                    msg.direction,
                    msg.rotation,
                    msg.time,
                    msg.steps,
                    interrupt=msg.interrupt,
                    relative_height=msg.relative_height,
                )
            elif msg.gait == 0:
                self.controller.stop_running(
                    timeout=None,
                    callback=lambda: self.controller.set_pose(None, None, msg.time),
                )
            elif msg.gait == -1:
                self.controller.stop_running()
            elif msg.gait == -2:
                self.controller.set_build_in_pose("DEFAULT_POSE", msg.time)
        except Exception as e:
            rospy.logerr(e)

    def set_leg_absolute_cb(self, leg_pos: jetmsg.LegPosition):
        pos = leg_pos.position.x, leg_pos.position.y, leg_pos.position.z
        self.controller.set_leg_position(leg_pos.leg_id, pos, leg_pos.duration)

    def set_leg_relatively_cb(self, leg_pos: jetmsg.LegPosition):
        offset = leg_pos.position.x, leg_pos.position.y, leg_pos.position.z
        cur_pos = list(self.controller.pose[leg_pos.leg_id - 1])
        new_pos = cur_pos[0] + offset[0], cur_pos[1] + offset[1], cur_pos[2] + offset[2]
        self.controller.set_leg_position(leg_pos.leg_id, new_pos, leg_pos.duration)

    def set_pose1_cb(self, req: SetPose1Request):
        rsp = SetPose1Response()
        try:
            self.controller.set_build_in_pose(req.pose, req.duration)
        except Exception as e:
            (rospy.logerr(e), setattr(rsp, "result", -1), setattr(rsp, "msg", str(e)))
        return rsp

    def set_pose2_cb(self, req: SetPose2Request):
        rsp = SetPose2Response()
        try:
            pose = [(point.x, point.y, point.z) for point in req.pose]
            self.controller.set_pose(pose, req.duration, interrupt=req.interrupt)
        except Exception as e:
            (rospy.logerr(e), setattr(rsp, "result", -1), setattr(rsp, "msg", str(e)))
        return rsp

    def set_pose_euler_cb(self, msg: jetmsg.Pose):
        self.controller.transform_absolutely(
            (msg.position.x, msg.position.y, msg.position.z),
            (msg.orientation.roll, msg.orientation.pitch, msg.orientation.yaw),
            0.4,
        )

    def pose_transform_euler_cb(self, msg: jetmsg.TransformEuler):
        try:
            self.controller.transform_pose_2(
                (msg.translation.x, msg.translation.y, msg.translation.z),
                "xyz",
                (msg.rotation.x, msg.rotation.y, msg.rotation.z),
                msg.duration,
                degrees=False,
            )
        except Exception as e:
            rospy.logerr(e)

    def run_action_set_sub_cb(self, msg: jetmsg.RunActionSet):
        file_path = (
            "/home/hiwonder/ActionSets/" + msg.action_path
            if msg.default_path
            else msg.action_path
        )
        self.controller.run_action_set(file_path, msg.repeat)


def main():
    jethexa_controller_node = jethexaControlNode("jethexa_control")
    try:
        rospy.spin()
    except Exception as e:
        rospy.logerr(e)


if __name__ == "__main__":
    main()
