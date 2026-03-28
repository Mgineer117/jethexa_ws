# JetHexa Hardware Integration Guide

**Minjae Cho** · Aerospace Engineering, UIUC · March 2026

---

## Overview

The JetHexa robot ships with pre-installed software and a custom ROS environment developed by Hiwonder. This default configuration facilitates immediate, out-of-the-box operation via user-friendly interfaces, such as the NoMachine remote desktop application and a proprietary mobile app.

While this configuration is advantageous for high-level control — where the system internally handles the inverse kinematics solvers — it restricts direct, low-level communication between a centralized Linux computer and the robot's hardware. Establishing a manual, bidirectional communication framework requires several system modifications. The following section outlines this setup procedure.

---

## 1. Setup

### 1.1 PC Setup

The centralized PC acts as the high-level controller. Clone this repository to the **home directory** to get started:
```bash
git clone https://github.com/Mgineer117/jethexa_ws
cd jethexa_ws
```

#### Communication Setup

Connect the PC to the robot's hotspot and export the network variables. Identify your local IP via `hostname -I`, then run:
```bash
export ROS_MASTER_URI=http://192.168.149.1:11311
export ROS_IP=<your_pc_ip>
```

---

### 1.2 Robot Setup

By default, the robot's operating system is configured to execute a `ros_bringup` script upon booting, which automatically initializes the core ROS nodes. However, this creates a significant network partition when attempting to bypass the manufacturer's recommended interfaces to command the robot from a centralized computer.

Specifically, the robot initializes its ROS master **before** establishing its Wi-Fi hotspot. Consequently, the ROS master binds to `localhost` (127.0.0.1) rather than the exposed network IP, effectively isolating the robot and preventing external machines from communicating with its ROS graph.

#### Step 1 — Disable Auto-Start Services

Disable the auto-starting behavior using the systemd service manager by executing the following on the robot's terminal:
```bash
sudo systemctl stop jethexa_bringup.service
sudo systemctl disable jethexa_bringup.service
```

These commands immediately terminate the active ROS background processes and prevent the daemon from initializing on subsequent boots. If necessary, this can be reversed by running:
```bash
sudo systemctl enable jethexa_bringup.service
```

or by reflashing the SD card with the original factory image.

#### Step 2 — Reconfigure Network Variables

The robot can be accessed physically via an attached monitor and keyboard, or remotely via SSH. To use SSH, connect to the robot's Wi-Fi hotspot and log in:

| Field        | Value                       |
|--------------|-----------------------------|
| SSID         | `HW_***`                    |
| Password     | `hiwonder`                  |
| SSH          | `ssh hiwonder@<gateway_ip>` |
| SSH Password | `hiwonder`                  |

Once connected, locate and edit the `.hiwonderrc` hidden file, which manages the ROS environment variables. To ensure the ROS master binds to the exposed Wi-Fi IP address rather than the local loopback, uncomment the following lines:
```bash
AUTO_ROS_HOSTNAME=true
AUTO_ROS_MASTER_URI=true
```

#### Step 3 — Manual ROS Bringup

After rebooting, the robot will no longer automatically start its ROS controllers — the leg servos will remain unpowered and the robot will not stand. To manually initialize the ROS environment and expose the hardware interface to the network, run:
```bash
roslaunch jethexa_bringup base.launch
```

Upon successful execution, the robot will engage its servos and assume its default standing posture, ready to receive external commands over the network.

#### Step 4 — Initialize Custom Joint Control

The original JetHexa framework only provides a high-level control interface. This repository introduces a new ROS topic enabling **independent per-joint control**. The relevant source files are located in:

- `src/jethexa_controller`
- `src/jethexa_controller_interfaces`

Run the provided initialization script to copy these files to the robot and trigger a ROS rebuild:
```bash
bash init.bash
```

The JetHexa's custom messages (e.g., `JointCommand`) are compiled for Python 2.7 by default. To allow Python 3 scripts to control the robot hardware, the compiled Python 3 message definitions must be manually moved into the active path:
```bash
# Define paths for the compiled Python 3 source and the active target directory
SOURCE_PY=~/jethexa/devel/.private/jethexa_controller_interfaces/lib/python3/dist-packages/jethexa_controller_interfaces/msg/_JointCommand.py
TARGET_DIR=~/jethexa/devel/lib/python2.7/dist-packages/jethexa_controller_interfaces/msg/

# Copy the definition and update the package initialization
cp $SOURCE_PY $TARGET_DIR/
echo "from ._JointCommand import *" >> $TARGET_DIR/__init__.py
```

---

## 2. Operation

### 2.1 Start the Robot

To initialize the system, launch the base controllers on the robot:
```bash
roslaunch jethexa_bringup base.launch
roslaunch jethexa_peripherals imu.launch
```

### 2.2 Verify Connection

Once the robot is standing, verify the connection from the PC by checking that the custom message type is recognized:
```bash
rosmsg show jethexa_controller_interfaces/JointCommand
```

If the terminal returns the `float32[18] target` structure, the bidirectional bridge is fully established.