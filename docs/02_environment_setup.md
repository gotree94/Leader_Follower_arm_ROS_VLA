# Environment Setup

> Leader-Follower Arm VLA 프로젝트의 전체 개발 환경 구성 가이드

---

## 1. NVIDIA Driver & CUDA 설치

### 1.1 NVIDIA Driver 570 설치 (RTX 5090)

```bash
# 기존 드라이버 제거
sudo apt purge nvidia-* -y
sudo apt autoremove -y

# NVIDIA Driver PPA 추가
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt update

# Driver 570 설치
sudo apt install nvidia-driver-570 nvidia-utils-570 -y

# 재부팅
sudo reboot
```

### 1.2 드라이버 확인

```bash
nvidia-smi
```

**예상 출력**:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 570.xx.xx    Driver Version: 570.xx.xx    CUDA Version: 12.6     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA RTX 5090     Off  | 00000000:01:00.0  On |                  Off |
| 30%   35C    P0    45W / 450W|      0MiB / 24564MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
```

### 1.3 CUDA 12.6 확인

```bash
nvcc --version
```

```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on ...
Cuda compilation tools, release 12.6, V12.6.xx
```

---

## 2. Docker + NVIDIA Container Toolkit

### 2.1 Docker 설치

```bash
# Docker GPG 키 추가
sudo apt update
sudo apt install ca-certificates curl gnupg -y
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

# Docker 저장소 추가
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt update

# Docker 설치
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin -y

# 사용자 권한 추가
sudo usermod -aG docker $USER
newgrp docker
```

### 2.2 NVIDIA Container Toolkit

```bash
# NVIDIA Container Toolkit 저장소 추가
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update

# 설치
sudo apt install nvidia-container-toolkit -y
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 확인
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi
```

### 2.3 Isaac Sim 2025.2 Docker 실행

```bash
# 이미지 Pull
docker pull nvcr.io/nvidia/isaac-sim:2025.2.0

# Isaac Sim 실행 스크립트
cat > ~/run_isaac_sim.sh << 'EOF'
#!/bin/bash
ISAAC_SIM_PATH="/home/$USER/.cache/isaac-sim"
mkdir -p $ISAAC_SIM_PATH

docker run --name isaac-sim -it --rm \
  --runtime=nvidia \
  -e ACCEPT_EULA=Y \
  -e PRIVACY_CONSENT=Y \
  -v $ISAAC_SIM_PATH/cache:/root/.cache:rw \
  -v $ISAAC_SIM_PATH/documents:/root/Documents:rw \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -e DISPLAY=$DISPLAY \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -p 9090:9090 \
  -p 9100:9100 \
  --network host \
  nvcr.io/nvidia/isaac-sim:2025.2.0
EOF
chmod +x ~/run_isaac_sim.sh
```

**usd_record 플러그인 활성화** (CosmosWriter용):

```bash
# Isaac Sim 내에서 Extension에서 "omni.writer.usd" 활성화
# 또는 startup script에 추가:
echo "import omni.writer.usd" > ~/.local/share/ov/data/Kit/Isaac-Sim/4.0/exts/isaac_sim_exts.toml
```

---

## 3. Isaac Lab 2.1 (Conda)

### 3.1 Conda 환경 생성

```bash
# Miniconda 설치 (없는 경우)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda init

# Isaac Lab 환경
conda create -n isaaclab python=3.10 -y
conda activate isaaclab
```

### 3.2 Isaac Lab 설치

```bash
# Isaac Lab 저장소 클론
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Isaac Lab 2.1 브랜치 체크아웃
git checkout v2.1.0

# pip 설치
pip install -e .

# 환경 확인
python -c "import isaaclab; print(isaaclab.__version__)"
```

### 3.3 VLA Extension 설치

```bash
# VLA 학습 확장 설치
cd source/extensions/omni.isaac.lab_tasks
pip install -e .

# 확인
python -c "from omni.isaac.lab_tasks.manager_based.manipulation.vla import vla_env; print('VLA extension OK')"
```

### 3.4 학습 동작 확인

```bash
# 간단한 테스트 학습 (Cartpole)
python scripts/reinforcement_learning/rl_games/play.py --task Isaac-Cartpole-Direct-v0

# GPU 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

---

## 4. Cosmos 2.0 (Conda)

### 4.1 Cosmos 환경 생성

```bash
# 별도 Conda 환경
conda create -n cosmos python=3.10 -y
conda activate cosmos
```

### 4.2 Cosmos-Reason 설치

```bash
# huggingface 로그인 (Cosmos 모델 접근용)
pip install huggingface-hub
huggingface-cli login
# → Hugging Face 토큰 입력 (nvidia-cosmos 모델 접근 권한 필요)

# Cosmos-Reason 설치
pip install cosmos-reason

# gRPC 서버 실행 확인
python -c "
from cosmos_reason import CosmosReasonClient
client = CosmosReasonClient()
print('Cosmos-Reason ready')
"
```

### 4.3 Cosmos-Policy 설치

```bash
# Cosmos-Policy 설치 (VLA 사전 학습)
pip install cosmos-policy

# 모델 다운로드 (약 12GB)
python -c "
from cosmos_policy import CosmosPolicy
policy = CosmosPolicy.from_pretrained('nvidia/cosmos-predict2-2b')
policy.save_pretrained('./models/cosmos-policy-base')
print('Cosmos-Policy downloaded')
"
```

### 4.4 Cosmos-Transfer 설치

```bash
# Sim-to-Real 이미지 변환
pip install cosmos-transfer

# 확인
python -c "
from cosmos_transfer import CosmosTransfer
print('Cosmos-Transfer ready')
"
```

### 4.5 API 키 설정

```bash
# 환경 변수 설정
export COSMOS_API_KEY="your-api-key-here"
export COSMOS_API_ENDPOINT="https://api.nvidia.com/v1/cosmos"

# .bashrc에 추가
echo 'export COSMOS_API_KEY="your-api-key-here"' >> ~/.bashrc
echo 'export COSMOS_API_ENDPOINT="https://api.nvidia.com/v1/cosmos"' >> ~/.bashrc
```

---

## 5. ROS2 Humble + 워크스페이스

### 5.1 ROS2 Humble 설치

```bash
# 로케일 설정
sudo apt update && sudo apt install locales -y
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

# ROS2 저장소 추가
sudo apt install software-properties-common -y
sudo add-apt-repository universe -y
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
sudo apt update

# ROS2 Humble 설치 (Desktop)
sudo apt install ros-humble-desktop python3-colcon-common-extensions -y

# 소스 설정
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

### 5.2 MoveIt2 설치

```bash
# MoveIt2 apt 설치
sudo apt install ros-humble-moveit ros-humble-moveit-ros-planning-interface \
  ros-humble-moveit-ros-visualization ros-humble-moveit-servo \
  ros-humble-moveit-hybrid-planning -y

# 확인
ros2 pkg list | grep moveit
```

### 5.3 Franka ROS2 드라이버

```bash
cd ~/ros2_ws/src
git clone https://github.com/frankaemika/franka_ros2.git -b foxy-devel
# Franka ROS2는 Foxy 브랜치이나 Humble과 호환
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select franka_msgs franka_gripper franka_robot_state
```

### 5.4 UR ROS2 드라이버

```bash
cd ~/ros2_ws/src
git clone https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver.git -b humble
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --packages-select ur_msgs ur_robot_driver ur_controllers
```

### 5.5 RealSense ROS2 드라이버

```bash
sudo apt install ros-humble-realsense2-camera -y
sudo apt install ros-humble-realsense2-description -y
```

### 5.6 VLA 인터페이스 패키지 생성

```bash
cd ~/ros2_ws/src
ros2 pkg create vla_interfaces --build-type ament_cmake --dependencies std_msgs geometry_msgs sensor_msgs

# 사용자 정의 메시지
cat > vla_interfaces/msg/VLAAction.msg << 'EOF'
# VLA 정책 추론 결과
std_msgs/Header header
float32[7] joint_velocities        # [6 follower joints + 1 gripper]
float32 success_probability        # 0.0 ~ 1.0
float32 task_progress               # 0.0 ~ 1.0
EOF

cat > vla_interfaces/msg/TaskPlan.msg << 'EOF'
# Cosmos Reason의 태스크 플랜
std_msgs/Header header
string task_id
string instruction                   # 원본 자연어 명령
string action_sequence[]             # ["approach", "grasp", "lift", ...]
float32[] target_poses               # flattened 6D poses
string[] object_names
EOF

# 빌드
cd ~/ros2_ws
colcon build --packages-select vla_interfaces
source install/setup.bash
```

---

## 6. DDS 네트워크 설정

### 6.1 멀티 머신 DDS 설정

Leader Arm, Follower Arm, Dev PC가 서로 다른 머신일 경우:

```bash
# Fast DDS 설정 파일
cat > ~/ros2_ws/fastdds_config.xml << 'EOF'
<?xml version="1.0" encoding="UTF-8" ?>
<profiles xmlns="http://www.eprosima.com/XMLSchemas/fastRTPS_Profiles">
    <transport_descriptors>
        <transport_descriptor>
            <transport_id>udp_transport</transport_id>
            <type>UDPv4</type>
            <maxMessageSize>65500</maxMessageSize>
            <sendBufferSize>1048576</sendBufferSize>
            <receiveBufferSize>1048576</receiveBufferSize>
        </transport_descriptor>
    </transport_descriptors>

    <participant profile_name="arm_robot_profile" is_default_profile="true">
        <rtps>
            <userTransports>
                <transport_id>udp_transport</transport_id>
            </userTransports>
            <useBuiltinTransports>false</useBuiltinTransports>
            <sendSocketBufferSize>1048576</sendSocketBufferSize>
            <listenSocketBufferSize>1048576</listenSocketBufferSize>
        </rtps>
    </participant>
</profiles>
EOF

export FASTRTPS_DEFAULT_PROFILES_FILE=~/ros2_ws/fastdds_config.xml
echo 'export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/ros2_ws/fastdds_config.xml' >> ~/.bashrc
```

### 6.2 QoS 프로파일 (실시간 Arm 제어)

```python
# qos_profiles.py
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

# Arm 제어용 QoS (Reliable + Transient Local)
ARM_CONTROL_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

# 카메라 영상용 QoS (Best Effort)
CAMERA_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=3,
)

# Digital Twin 데이터용 QoS (Reliable + Keep All)
DT_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_ALL,
    depth=100,
)
```

### 6.3 지연 테스트

```bash
# ROS2 latency test
ros2 run demo_nodes_cpp talker &
ros2 run demo_nodes_cpp listener &
# 확인: talker → listener 지연 < 1ms (동일 머신)
```

---

## 7. 전체 설치 검증

```bash
# 검증 스크립트
cat > ~/verify_setup.sh << 'EOF'
#!/bin/bash
echo "=== Setup Verification ==="

# 1. NVIDIA Driver
echo -n "NVIDIA Driver: "
nvidia-smi | grep -oP 'Driver Version: \K[0-9.]+'

# 2. Docker
echo -n "Docker: "
docker --version | cut -d' ' -f3 | tr -d ','

# 3. Isaac Sim Image
echo -n "Isaac Sim: "
docker images nvcr.io/nvidia/isaac-sim:2025.2.0 --format "{{.Tag}}"

# 4. Isaac Lab
echo -n "Isaac Lab: "
conda run -n isaaclab python -c "import isaaclab; print(isaaclab.__version__, end='')"

# 5. Cosmos
echo -n "Cosmos-Reason: "
conda run -n cosmos python -c "import cosmos_reason; print('OK', end='')"

# 6. ROS2
echo -n "ROS2: "
ros2 --version | cut -d' ' -f2

# 7. MoveIt2
echo -n "MoveIt2: "
ros2 pkg list | grep -c moveit

echo "=== Verification Complete ==="
EOF
chmod +x ~/verify_setup.sh
~/verify_setup.sh
```

---

## 8. 문제 해결

| 문제 | 원인 | 해결 |
|------|------|------|
| Docker GPU 접근 불가 | NVIDIA Container Toolkit 미설치 | `sudo apt install nvidia-container-toolkit` |
| Isaac Sim GUI 안 보임 | X11 권한 | `xhost +local:docker` |
| Conda Python 버전 불일치 | Isaac Sim은 Python 3.10 필요 | `conda create -n isaaclab python=3.10` |
| ROS2 노드 간 통신 안 됨 | DDS 설정 누락 | `export ROS_DOMAIN_ID=42` |
| MoveIt2 IK 실패 | URDF 충돌 메시지 누락 | `collision_checking_type: "geometric"` |
| Cosmos API 401 오류 | API 키 누락 | `export COSMOS_API_KEY="..."` |
| TensorRT 변환 실패 | ONNX opset 버전 | `opset_version=17` 사용 |
| Jetson 추론 느림 | 전원 모드 | `sudo nvpmodel -m 0` (MAXN) |
