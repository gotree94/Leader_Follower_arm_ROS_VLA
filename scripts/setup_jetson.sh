#!/bin/bash
# Jetson Orin Setup for Follower Arm VLA Inference
set -e

echo "=== Jetson Orin VLA Setup ==="

# MAXN Performance
sudo nvpmodel -m 0
sudo jetson_clocks

# Memory optimization
echo 2048 | sudo tee /proc/sys/vm/min_free_kbytes

# CUDA
export CUDA_MALLOC_ASYNC=1
export TRT_CACHEDIR=/tmp/trt_cache
mkdir -p $TRT_CACHEDIR

# ROS2
export ROS_DOMAIN_ID=42

echo "Jetson Orin ready for VLA inference"
