#!/bin/bash
# Leader-Follower Arm VLA Environment Setup Script

set -e

echo "=== Arm VLA Environment Setup ==="

# 1. System Dependencies
echo "[1/6] Installing system dependencies..."
sudo apt update
sudo apt install -y python3-pip python3-venv curl wget git build-essential

# 2. ROS2 Humble + MoveIt2
echo "[2/6] Installing ROS2 Humble + MoveIt2..."
sudo apt install -y ros-humble-desktop \
    ros-humble-moveit \
    ros-humble-moveit-ros-planning-interface \
    ros-humble-moveit-servo \
    ros-humble-realsense2-camera \
    python3-colcon-common-extensions

# 3. Isaac Lab Conda Environment
echo "[3/6] Setting up Isaac Lab environment..."
conda create -n isaaclab python=3.10 -y
source $(conda info --base)/etc/profile.d/conda.sh
conda activate isaaclab
git clone https://github.com/isaac-sim/IsaacLab.git /tmp/IsaacLab
cd /tmp/IsaacLab && git checkout v2.1.0 && pip install -e .
cd source/extensions/omni.isaac.lab_tasks && pip install -e .
cd /workspace

# 4. Cosmos 2.0 Environment
echo "[4/6] Setting up Cosmos environment..."
conda create -n cosmos python=3.10 -y
conda activate cosmos
pip install cosmos-reason cosmos-policy cosmos-transfer

# 5. Create Working Directory
echo "[5/6] Creating workspace..."
mkdir -p logs models results data

# 6. Verification
echo "[6/6] Verifying installation..."
echo "Environment setup complete."
echo ""
echo "Available Conda environments:"
conda env list
