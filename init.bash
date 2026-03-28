#!/bin/bash

# --- Configuration ---
ROBOT_IP="192.168.149.1"
ROBOT_USER="hiwonder"
REMOTE_WS="/home/hiwonder/jethexa"
REMOTE_ADDR="$ROBOT_USER@$ROBOT_IP"

# --- Dynamic Local Path Discovery ---
# Scoping the search to the specific workspace source folder for speed and accuracy
LOCAL_WS="$HOME/jethexa_ws/src"

LOCAL_INTF_DIR=$(find "$LOCAL_WS" -type d -name "jethexa_controller_interfaces" -print -quit)
LOCAL_PY=$(find "$LOCAL_WS" -type f -name "jethexa_controller_main.py" -print -quit)

# Validation: Stop if files aren't found
if [ -z "$LOCAL_INTF_DIR" ] || [ -z "$LOCAL_PY" ]; then
    echo "ERROR: Could not locate required source files within $LOCAL_WS."
    echo "Check if 'jethexa_controller_interfaces' and 'jethexa_controller_main.py' exist there."
    exit 1
fi

# Define sub-paths based on discovered directory
LOCAL_MSG="$LOCAL_INTF_DIR/msg/JointCommand.msg"
LOCAL_CMAKE="$LOCAL_INTF_DIR/CMakeLists.txt"
LOCAL_XML="$LOCAL_INTF_DIR/package.xml"

# --- Remote Destination Paths ---
REMOTE_INTF_DEST="$REMOTE_WS/src/jethexa_controller/jethexa_controller_interfaces"
REMOTE_PY_DEST="$REMOTE_WS/src/jethexa_controller/jethexa_controller/scripts/jethexa_controller_main.py"

echo "--- Starting Scoped Deployment: $LOCAL_WS -> $REMOTE_ADDR ---"

# 1. Sync Interface Package (Messages & Build Config)
echo "Syncing message definitions..."
rsync -avzc --inplace --chmod=644 "$LOCAL_CMAKE" "$REMOTE_ADDR:$REMOTE_INTF_DEST/CMakeLists.txt"
rsync -avzc --inplace --chmod=644 "$LOCAL_XML" "$REMOTE_ADDR:$REMOTE_INTF_DEST/package.xml"
rsync -avzc --inplace "$LOCAL_MSG" "$REMOTE_ADDR:$REMOTE_INTF_DEST/msg/JointCommand.msg"

# 2. Sync Controller Script
echo "Syncing controller script..."
rsync -avzc "$LOCAL_PY" "$REMOTE_ADDR:$REMOTE_PY_DEST"
ssh "$REMOTE_ADDR" "chmod +x $REMOTE_PY_DEST"

# 3. Remote Build and Python 3 Patch
echo "Triggering remote build and applying Python 3 compatibility patch..."
ssh "$REMOTE_ADDR" << EOF
    source /opt/ros/melodic/setup.sh
    cd "$REMOTE_WS"
    
    # Force CMake to re-index the package in case only msg content changed
    touch "$REMOTE_INTF_DEST/CMakeLists.txt"
    
    # Build only the necessary packages to save time
    catkin build jethexa_controller_interfaces jethexa_controller
    
    source devel/setup.sh
    rospack profile

    # Define Patch Paths (Internal to Robot)
    # This addresses the Melodic/Python 3 bridge requirement for the JetHexa
    TARGET_DIR="$REMOTE_WS/devel/lib/python2.7/dist-packages/jethexa_controller_interfaces/msg"
    SOURCE_PY="$REMOTE_WS/devel/.private/jethexa_controller_interfaces/lib/python3/dist-packages/jethexa_controller_interfaces/msg/_JointCommand.py"

    if [ -f "\$SOURCE_PY" ]; then
        cp "\$SOURCE_PY" "\$TARGET_DIR/"
        # Append import only if not already present
        if ! grep -q "from ._JointCommand import *" "\$TARGET_DIR/__init__.py"; then
            echo "from ._JointCommand import *" >> "\$TARGET_DIR/__init__.py"
        fi
        echo "SUCCESS: Python 3 message patch applied."
    else
        echo "WARNING: Compiled Python 3 message not found at \$SOURCE_PY"
    fi
EOF

echo "--- Deployment Finished Successfully ---"