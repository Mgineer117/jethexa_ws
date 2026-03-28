# JetHexa Hardware Integration Guide

**Minjae Cho**  
*Aerospace Engineering, UIUC*  
March 2026  

---

## Setup

The robot is supplied with pre-installed software and a custom ROS environment developed by the manufacturer, Hiwonder. While this facilitates out-of-the-box operation, establishing a manual, bidirectional communication framework for low-level research requires specific system modifications on both the robot and the centralized PC.

---

## Robot Setup

By default, the robot's ROS master binds to `localhost`, isolating it from the network. After disabling the `jethexa_bringup` service and configuring `.hiwonderrc` as previously detailed, a critical Python compatibility patch is required.

### Python 3 Message Patch

The JetHexa's custom messages (e.g., `JointCommand`) are compiled for Python 2.7 by default. To allow Python 3 scripts to control the robot hardware locally or remotely, the compiled Python 3 message definitions must be manually moved into the active path:

```bash
# Define paths for the compiled Python 3 source and the active target directory
SOURCE_PY=~/jethexa/devel/.private/jethexa_controller_interfaces/lib/python3/dist-packages/jethexa_controller_interfaces/msg/_JointCommand.py
TARGET_DIR=~/jethexa/devel/lib/python2.7/dist-packages/jethexa_controller_interfaces/msg/

# Copy the definition and update the package initialization
cp $SOURCE_PY $TARGET_DIR/
echo "from ._JointCommand import *" >> $TARGET_DIR/__init__.py

## PC Setup

The centralized PC acts as the high-level controller. To communicate with the robot, it must possess the "blueprints" for the JetHexa's custom ROS messages.

### Workspace Initialization

On the PC, we must mirror the robot's interface definitions. If the `src` folder is empty, the PC will not recognize custom types like `JointCommand`. Execute the following to initialize the workspace and pull the definitions from the robot:

```bash
# 1. Create structure
mkdir -p ~/jethexa_ws/src
cd ~/jethexa_ws/src

# 2. Pull message definitions from the robot (Remote Mirroring)
scp -r hiwonder@192.168.149.1:~/jethexa/src/jethexa_controller/jethexa_controller_interfaces .

# 3. Build the local 'blueprints' on the PC
cd ~/jethexa_ws
catkin_make -DCATKIN_WHITELIST_PACKAGES="jethexa_controller_interfaces"
source devel/setup.bash