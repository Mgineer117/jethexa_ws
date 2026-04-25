# JetHexa Robot Setup and Operation Guide

This repository contains the PC-side workspace and custom ROS interfaces used to control a JetHexa robot from an external Linux machine. It also includes data collection, trajectory playback, and motion-capture integration utilities.

The goal of this guide is to document the setup in the order it should be completed, explain the purpose of the important files, and make recovery easier when the robot image becomes corrupted.

## Overview

The default JetHexa software image is designed for the manufacturer's workflow. That works well for basic operation, but it is not ideal when you want a separate Linux PC to act as the main controller and send low-level joint commands over ROS.

This repository adds that workflow by:

1. Building a ROS workspace on the PC.
2. Reconfiguring the robot so its ROS master is reachable over Wi-Fi.
3. Installing the custom `JointCommand` interface and updated controller code onto the robot.
4. Running playback, collection, and control scripts from the PC.

## System Layout

- PC: runs this repository, user scripts, trajectory tools, and optionally the Qualisys bridge.
- Robot: runs the JetHexa ROS stack and executes joint commands.
- Qualisys motion capture system: optional, but required for scripts that use torso pose feedback from mocap.

## Setup Order

Complete the following sections in order.

### 1. Clone the Workspace on the PC

```bash
git clone https://github.com/Mgineer117/jethexa_ws
cd jethexa_ws
```

### 2. Install Conda on the PC

Skip this section if Conda is already installed.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

### 3. Create the Software Environment

Create the Conda environment and install the ROS and Python dependencies used by this workspace.

```bash
conda create -n jethexa python=3.9.18 -y
conda activate jethexa
conda install ros-noetic-ros-base ros-noetic-catkin catkin_tools -c robostack-staging -c conda-forge
pip install scipy qtm-rt matplotlib numpy torch
```

Verify the installation:

```bash
roscore -h
python3 -c "import rospy; print('ROS Python Bridge: Success')"
```

### 4. Build the PC Workspace

From the repository root:

```bash
conda activate jethexa
catkin_make -DCMAKE_POLICY_VERSION_MINIMUM=3.5
source devel/setup.bash
```

### 5. Connect the PC to the Robot Network

Connect the PC to the robot hotspot. The JetHexa hotspot usually looks like `HW_***` and the default password is `hiwonder`.

Set the ROS networking variables on the PC:

```bash
export ROS_MASTER_URI=http://192.168.149.1:11311
export ROS_IP=<your_pc_ip>
```

Then source the environment:

```bash
source ~/.bashrc
conda activate jethexa
source devel/setup.bash
```

You can get the PC IP with:

```bash
hostname -I
```

## Robot Setup

These steps configure the robot so the ROS graph is reachable from the PC.

### 6. Disable the Default Auto-Start Service

By default the robot launches its ROS stack too early during boot, before networking is fully ready. That can cause the ROS master to bind incorrectly and become unreachable from the PC.

Run these commands on the robot:

```bash
sudo systemctl stop jethexa_bringup.service
sudo systemctl disable jethexa_bringup.service
```

If you ever need to restore the original behavior:

```bash
sudo systemctl enable jethexa_bringup.service
```

### 7. Fix the Robot ROS Networking Configuration

SSH into the robot after connecting to its hotspot:

```bash
ssh hiwonder@192.168.149.1
```

Default credentials:

- Username: `hiwonder`
- Password: `hiwonder`

Edit the robot's hidden ROS environment configuration and enable automatic hostname/master URI setup in `.hiwonderrc`:

```bash
AUTO_ROS_HOSTNAME=true
AUTO_ROS_MASTER_URI=true
```

After changing this file, reboot the robot or restart the shell session so the updated configuration takes effect.

### 8. Manually Start the Robot ROS Stack

After disabling the service, the robot will not stand up automatically at boot. Start its base stack manually:

```bash
roslaunch jethexa_bringup base.launch
```

If your workflow uses the IMU, also run:

```bash
roslaunch jethexa_peripherals imu.launch
```

At this point the robot should power the servos and move into its default standing posture.

### 9. Deploy the Custom Interface and Controller

This repository includes a custom `JointCommand` message and modified controller code so the PC can publish direct joint targets.

Run:

```bash
bash init.bash
```

This script is important and does several things in order:

1. Assumes the local workspace source tree is at `$HOME/jethexa_ws/src`.
2. Finds the local `jethexa_controller_interfaces` package.
3. Finds the local `jethexa_controller_main.py` controller script.
4. Copies `JointCommand.msg`, `CMakeLists.txt`, and `package.xml` to the robot.
5. Copies the updated controller script to the robot.
6. SSHes into the robot and rebuilds the relevant ROS packages with `catkin build`.
7. Refreshes the ROS package index with `rospack profile`.
8. Copies the generated Python 3 `_JointCommand.py` file into the robot's active Python 2.7 ROS message path.
9. Appends `from ._JointCommand import *` to the target `__init__.py` if needed.

That last patch is required because the robot image is based on ROS Melodic and the default message path used by the running system is still tied to Python 2.7, while these scripts run in Python 3 on the PC side.

Important note: `init.bash` currently assumes the repository lives at `~/jethexa_ws` on the PC. If your local workspace is elsewhere, update the `LOCAL_WS` path inside `init.bash` before running it.

### 10. Verify the Custom Message is Visible

From the PC:

```bash
rosmsg show jethexa_controller_interfaces/JointCommand
```

If it returns the `target` array and `duration` field, the message bridge is working.

## Daily Startup Order

Once the one-time setup is finished, this is the normal bring-up sequence.

### Basic robot bring-up

On the robot:

```bash
roslaunch jethexa_bringup base.launch
roslaunch jethexa_peripherals imu.launch
```

On the PC:

```bash
conda activate jethexa
source devel/setup.bash
export ROS_MASTER_URI=http://192.168.149.1:11311
export ROS_IP=<your_pc_ip>
```

### If using Qualisys mocap

Start the Qualisys bridge in a separate terminal on the PC:

```bash
conda activate jethexa
source devel/setup.bash
python3 qualysis.py --marker_deck_name jethexa
```

This must be running before any mocap-dependent script can receive pose updates on the `qualysis/<marker_deck_name>` topic. If `qualysis.py` is not running, scripts that require motion-capture data will wait for that stream and never become ready.

In this repository, `generate_trajectory.py` and `control_loop.py` both depend on that pose stream. `play_abs_traj.py` and `play_rel_traj.py` do not.

### Then run the desired application

Examples:

```bash
python3 play_abs_traj.py
python3 play_rel_traj.py
python3 generate_trajectory.py --mode turning --duration 33
python3 control_loop.py --algo-name ppo --control-scaler 0.3 --duration 33
```

## Recovery and Reflashing

Sometimes the robot image gets corrupted and the robot will no longer boot or behave correctly. In practice people often describe this as the robot firmware being broken, even though the recovery is usually just restoring the SD card image.

The recovery procedure is straightforward:

1. Power off the robot.
2. Remove the SD card from the robot.
3. Format the SD card.
4. Reinstall the operating system image provided by the manufacturer.
5. Reinsert the SD card and boot the robot.
6. Repeat the robot setup steps in this README.

After reflashing, assume the robot has returned to the factory state. That means you will usually need to:

1. Disable `jethexa_bringup.service` again.
2. Reapply the `.hiwonderrc` network configuration.
3. Manually start the robot ROS stack.
4. Run `bash init.bash` again to reinstall the custom message and controller changes.

## File Guide

This section explains the main top-level files used in the workflow.

### `init.bash`

Deployment script that pushes the custom ROS interface and controller code from the PC to the robot, rebuilds the robot workspace, and applies the Python 3 message compatibility patch.

### `base.py`

Shared support class used by the playback and control scripts. It:

- subscribes to joint states, IMU, and mocap topics depending on configuration
- publishes absolute or relative joint commands
- loads trajectory files
- waits for required sensors to become ready
- checks joint limits
- moves the robot to the initial pose before playback
- returns the robot to the default pose at the end

### `qualysis.py`

Bridge between the Qualisys motion-capture system and ROS. It connects to the Qualisys server, reads the rigid-body pose, and publishes it as a ROS `PoseStamped` message on:

```bash
qualysis/<marker_deck_name>
```

Use this in a separate terminal whenever a script depends on mocap feedback. For example, `control_loop.py` enables `use_mocap=True`, so `qualysis.py` must already be running or the script will wait for that sensor stream.

### `generate_trajectory.py`

Collects robot motion data while commanding built-in velocity motions. It records:

- base position and orientation
- joint positions
- estimated joint controls

It saves the result to `hexapod_data/*.npz` and also writes a trajectory plot as a `.png`.

This script subscribes to `/qualysis/jethexa` for torso pose, so run `qualysis.py` first in a separate terminal.

Available collection modes:

- `turning`
- `accel`
- `combined`

### `analyze_trajectories.py`

Loads the most recent `.npz` file in `hexapod_data/` and plots:

- torso position
- torso orientation
- all 18 joint trajectories

Use this after data collection to inspect whether the recorded motion looks reasonable.

### `play_abs_traj.py`

Loads the test trajectory from `models/test_traj.npz`, extracts the absolute joint positions from the state vector, and replays them directly as 18-DOF joint targets. This is the most literal playback of a stored trajectory.

### `play_rel_traj.py`

Loads a stored trajectory, reads the `controls` array, integrates those controls over time, and converts them into virtual absolute joint targets before publishing them. Use this when you want playback based on the recorded joint increments/velocities rather than directly replaying stored absolute angles.

### `control_loop.py`

Runs a learned policy during live execution. It:

- loads a policy network from `models/`
- loads a reference trajectory from `models/test_traj.npz`
- reads the current robot state
- computes corrective joint actions
- publishes the resulting absolute joint targets
- saves the rollout back into `hexapod_data/`

Because its configuration sets `use_mocap=True`, this script depends on the `qualysis.py` bridge being active if you want the base pose stream to be available.

## ROS Packages Added by This Repository

### `src/jethexa_controller_interfaces`

Defines the custom ROS messages and services, including `JointCommand.msg`.

### `src/jethexa_controller`

Contains the custom controller logic and scripts used to expose lower-level joint control on the robot.

### `src/jethexa_sdk`

Contains the JetHexa SDK support code used by the controller stack.

## Typical Commands

Collect a trajectory:

```bash
python3 generate_trajectory.py --mode turning --duration 33
```

Analyze the newest trajectory:

```bash
python3 analyze_trajectories.py
```

Replay a stored absolute trajectory:

```bash
python3 play_abs_traj.py
```

Replay a stored relative/control-based trajectory:

```bash
python3 play_rel_traj.py
```

Run the learned controller:

```bash
python3 control_loop.py --algo-name ppo --control-scaler 0.3 --duration 33
```
