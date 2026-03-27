# JetHexa ROS Environment Setup

This guide outlines the steps to configure an isolated ROS Noetic environment using Conda (via RoboStack) and initialize the catkin workspace for the JetHexa project.

## 1. Installing ROS (Environment Setup)

Once Conda is installed on your system, create and activate a dedicated environment for the JetHexa project. Then, configure the required channels for RoboStack before installing the dependencies:

```bash
# Create and activate the conda environment
conda create -n jethexa python=3.9.18 -y
conda activate jethexa

# Configure the channels specifically for this environment
conda config --env --add channels conda-forge
conda config --env --add channels robostack-staging
conda config --env --remove channels defaults

# Install essential ROS desktop and necessary build tools
conda install ros-noetic-desktop compilers cmake pkg-config make ninja catkin_tools rospkg -y
```

This setup ensures that your Python virtual environment is isolated and pre-configured with the necessary ROS libraries. 

## 2. Verify Installation

To verify that the environment is correctly sourcing ROS and recognizes the Python interpreter, run the following verification commands:

```bash
roscore -h
python3 -c "import rospy; print('ROS Python Bridge: Success')"
```

**Note:** If the first command returns the help manual for the ROS Master and the second prints the success message, your PC is ready to interface with the robot.

## 3. Workspace Initialization

Finally, create the workspace directory structure, build the workspace, and source the setup file so ROS can find your packages:

```bash
# Create the workspace directory structure
mkdir -p ~/jethexa/src
cd ~/jethexa

# Initialize and build the workspace
catkin_make

# Source the setup file to apply the workspace overlay
source devel/setup.bash
```

---

*Would you like me to add a section to this README explaining how to run the `debug_joints_relative.py` script we created earlier?*