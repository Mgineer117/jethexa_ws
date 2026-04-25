#!/usr/bin/env python3
# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Qualisys -> ROS pose bridge.

Connects to a Qualisys motion-capture server over qtm_rt, subscribes to a
named rigid body (`--marker_deck_name`), and republishes its 6-DOF pose
as a geometry_msgs/PoseStamped on /qualysis/<marker_deck_name>.

Any script in this repo that needs torso pose feedback (control_loop.py,
generate_trajectory.py, dynamics_audit.py) listens on that topic, so this
bridge has to be running first or those scripts will block waiting for
the stream.

Adapted from
https://github.com/tbretl/crazyflie-client/blob/main/ae483clients.py
"""

import argparse
import asyncio
import os
import sys
import xml.etree.cElementTree as ET
from pathlib import Path
from threading import Thread

# Keep cwd consistent with the rest of the scripts in this folder.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.chdir(_REPO_ROOT)

import numpy as np
import qtm_rt as qtm
import rospy
from geometry_msgs.msg import PoseStamped
from scipy.spatial.transform import Rotation


class QualysisPublisher(object):
    def __init__(self, ip_address, marker_deck_name):
        self.ip_address = ip_address
        self.marker_deck_name = marker_deck_name

        # ROS 1 Publisher with queue_size instead of QoS Profile
        self.publisher_ = rospy.Publisher(
            f"qualysis/{marker_deck_name}", PoseStamped, queue_size=1
        )

        # Initialize Qualisys Client thread
        self.qualysis_client = QualisysClient(self.ip_address, self.marker_deck_name)

        # ROS 1 Timer
        timer_period = 0.01  # seconds
        self.timer = rospy.Timer(rospy.Duration(timer_period), self.timer_callback)

    def timer_callback(self, event=None):
        pose = PoseStamped()

        data = self.qualysis_client.data

        pose.header.frame_id = "mocap"
        pose.header.stamp = rospy.Time.now()  # ROS 1 time format

        pose.pose.position.x = data["x"] if data["x"] else 0.0
        pose.pose.position.y = data["y"] if data["y"] else 0.0
        pose.pose.position.z = data["z"] if data["z"] else 0.0

        # Note: standard ROS expects quaternions here, but maintaining
        # original logic of passing Euler angles into quaternion fields.
        # quat = data['quaternion']
        # pose.pose.orientation.x = quat[0] if quat[0] else 0.0
        # pose.pose.orientation.y = quat[1] if quat[1] else 0.0
        # pose.pose.orientation.z = quat[2] if quat[2] else 0.0
        # pose.pose.orientation.w = quat[3] if quat[3] else 0.0

        pose.pose.orientation.z = data["yaw"] if data["yaw"] else 0.0
        pose.pose.orientation.y = data["pitch"] if data["pitch"] else 0.0
        pose.pose.orientation.x = data["roll"] if data["roll"] else 0.0
        pose.pose.orientation.w = 0.0

        self.publisher_.publish(pose)


class QualisysClient(Thread):
    def __init__(self, ip_address, marker_deck_name):
        Thread.__init__(self)
        self.ip_address = ip_address
        self.marker_deck_name = marker_deck_name
        self.connection = None
        self.qtm_6DoF_labels = []
        self._stay_open = True
        self.data = {
            "time": [],
            "x": [],
            "y": [],
            "z": [],
            # 'quaternion': []
            "yaw": [],
            "pitch": [],
            "roll": [],
        }
        self.start()

    def close(self):
        self._stay_open = False
        self.join()

    def run(self):
        # Note: asyncio.run requires Python 3.7+.
        # If your Melodic setup uses Python 3.6, change this to:
        # loop = asyncio.new_event_loop()
        # loop.run_until_complete(self._life_cycle())
        asyncio.run(self._life_cycle())

    async def _life_cycle(self):
        await self._connect()
        while self._stay_open:
            await asyncio.sleep(1)
        await self._close()

    async def _connect(self):
        print("QualisysClient: Connect to motion capture system")
        self.connection = await qtm.connect(self.ip_address, version="1.24")
        params = await self.connection.get_parameters(parameters=["6d"])
        xml = ET.fromstring(params)
        self.qtm_6DoF_labels = [
            label.text.strip() for index, label in enumerate(xml.findall("*/Body/Name"))
        ]
        await self.connection.stream_frames(
            components=["6d"],
            on_packet=self._on_packet,
        )

    def _on_packet(self, packet):
        header, bodies = packet.get_6d()

        if bodies is None:
            print("QualisysClient: No rigid bodies found")
            return

        if self.marker_deck_name not in self.qtm_6DoF_labels:
            print(f"QualisysClient: Marker deck {self.marker_deck_name} not found")
            return

        index = self.qtm_6DoF_labels.index(self.marker_deck_name)
        position, orientation = bodies[index]

        # Get time in seconds, with respect to the qualisys clock
        t = packet.timestamp / 1e6

        # Get position of marker deck (x, y, z in meters)
        x, y, z = np.array(position) / 1e3

        # Get orientation of marker deck (yaw, pitch, roll in radians)
        R = Rotation.from_matrix(np.reshape(orientation.matrix, (3, -1), order="F"))
        # quaternion = R.as_quat()#euler('ZYX', degrees=False)
        yaw, pitch, roll = R.as_euler("ZYX", degrees=False)

        # Store time, position, and orientation
        self.data["time"] = t
        self.data["x"] = x
        self.data["y"] = y
        self.data["z"] = z
        # self.data['quaternion'] = quaternion
        self.data["yaw"] = yaw
        self.data["pitch"] = pitch
        self.data["roll"] = roll

    def get_pose(self, packet):
        header, bodies = packet.get_6d()

        if bodies is None:
            print("QualisysClient: No rigid bodies found")
            return

        if self.marker_deck_name not in self.qtm_6DoF_labels:
            print(f"QualisysClient: Marker deck {self.marker_deck_name} not found")
            return

        index = self.qtm_6DoF_labels.index(self.marker_deck_name)
        position, orientation = bodies[index]

        # Get time in seconds, with respect to the qualisys clock
        t = packet.timestamp / 1e6

        # Get position of marker deck (x, y, z in meters)
        x, y, z = np.array(position) / 1e3

        # Get orientation of marker deck (yaw, pitch, roll in radians)
        R = Rotation.from_matrix(np.reshape(orientation.matrix, (3, -1), order="F"))
        # quaternion = R.as_quat()#euler('ZYX', degrees=False)
        yaw, pitch, roll = R.as_euler("ZYX", degrees=False)

        return np.array([x, y, z, yaw, pitch, roll])

    async def _close(self):
        await self.connection.stream_frames_stop()
        self.connection.disconnect()


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Please enter the marker deck name of the rigid body defined in the qualysis system"
    )
    parser.add_argument("--marker_deck_name", required=True, type=str)
    # Using parse_known_args() so ROS-specific args (like __name:=...) don't crash argparse
    p_args, unknown = parser.parse_known_args()

    # IP address of the motion capture system
    ip_address = "128.174.245.64"
    marker_deck_name = p_args.marker_deck_name

    # Initialize ROS 1 Node
    rospy.init_node("qualysis_publisher", anonymous=True)

    qualysis_publisher = QualysisPublisher(ip_address, marker_deck_name)

    # Safely close the Asyncio thread when ROS shuts down (e.g., Ctrl+C)
    rospy.on_shutdown(qualysis_publisher.qualysis_client.close)

    # Spin to keep the script from exiting
    rospy.spin()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
