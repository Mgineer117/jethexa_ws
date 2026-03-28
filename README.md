JetHexa Hardware Integration Guide
Author: Minjae Cho

Aerospace Engineering, UIUC Date: March 2026

Overview
The Hiwonder JetHexa comes pre-installed with a custom ROS environment. While this works for basic operations, low-level research requires establishing a manual, bidirectional communication framework between the robot and a centralized PC. This guide details the modifications necessary to enable Python 3 compatibility and remote message mirroring.

1. Setup
1.1 Robot Setup
By default, the JetHexa's ROS master binds to localhost. To integrate it into a wider network, you must first disable the jethexa_bringup service and configure .hiwonderrc as previously established.

Python 3 Message Patch
The robot's custom messages (such as JointCommand) are compiled for Python 2.7 by default. To allow Python 3 scripts—common in modern reinforcement learning and robotics stacks—to control the hardware, you must manually move the compiled Python 3 definitions into the active path.

Apply the patch:

Bash
# Define paths for the compiled Python 3 source and the active target directory
SOURCE_PY=~/jethexa/devel/.private/jethexa_controller_interfaces/lib/python3/dist-packages/jethexa_controller_interfaces/msg/_JointCommand.py
TARGET_DIR=~/jethexa/devel/lib/python2.7/dist-packages/jethexa_controller_interfaces/msg/

# Copy the definition and update the package initialization
cp $SOURCE_PY $TARGET_DIR/
echo "from ._JointCommand import *" >> $TARGET_DIR/__init__.py
1.2 PC Setup
The centralized PC serves as the high-level controller. For the PC to communicate with the robot, it must possess the "blueprints" for the JetHexa’s custom ROS messages.

Workspace Initialization
If your local src folder is empty, the PC will not recognize custom types like JointCommand. You must mirror the interface definitions from the robot:

Bash
# 1. Create structure
mkdir -p ~/jethexa_ws/src
cd ~/jethexa_ws/src

# 2. Pull message definitions from the robot (Remote Mirroring)
# Ensure you are connected to the robot's network
scp -r hiwonder@192.168.149.1:~/jethexa/src/jethexa_controller/jethexa_controller_interfaces .

# 3. Build the local 'blueprints' on the PC
cd ~/jethexa_ws
catkin_make -DCATKIN_WHITELIST_PACKAGES="jethexa_controller_interfaces"
source devel/setup.bash
Communication Setup
Connect your PC to the JetHexa hotspot. Identify your local IP address using hostname -I and export the following network variables:

Bash
export ROS_MASTER_URI=http://192.168.149.1:11311
export ROS_IP=<your_pc_ip>
2. Operation
2.1 Manual Robot Bringup
To initialize the system, SSH into the robot and start the base controllers and peripherals:

Bash
# Start the base controllers
roslaunch jethexa_bringup base.launch

# Start the IMU and other peripherals
roslaunch jethexa_peripherals imu.launch
2.2 Verify Connection
Once the robot is in a standing posture, verify that the PC correctly recognizes the custom message types:

Bash
rosmsg show jethexa_controller_interfaces/JointCommand
Expected Output:
If the terminal returns the float32[18] target structure, the bidirectional bridge is fully established and ready for low-level control.

Would you like me to generate a Python 3 template script that utilizes these JointCommand messages to test the hexapod's leg movements?