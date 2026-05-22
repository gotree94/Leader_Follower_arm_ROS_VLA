# System Requirements & Prerequisites

> Leader-Follower Arm VLA 프로젝트의 하드웨어/소프트웨어 요구사항

---

## 1. 하드웨어 요구사항

### 1.1 Development PC (필수)

| 구성 요소 | Minimum | Recommended | 비고 |
|-----------|---------|-------------|------|
| **GPU** | RTX 4090 24GB | **RTX 5090 24GB** | VLA Transformer 학습 핵심 장비 |
| **VRAM** | 24GB | 32GB 이상 | batch_size=128+ 학습 시 필요 |
| **RAM** | 32GB | **64GB** | Isaac Sim + Isaac Lab 동시 실행 시 필수 |
| **CPU** | Intel i7 / AMD Ryzen 7 (8코어) | Intel Ultra 9 (16코어) | 병렬 시뮬레이션 환경 |
| **Storage** | NVMe SSD 500GB | NVMe SSD 2TB+ | 에피소드 데이터 + 체크포인트 |
| **OS** | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS | |

### 1.2 Robot Arms

| 역할 | 권장 로봇 | DOF | 특징 |
|------|-----------|-----|------|
| **Leader Arm** (시연) | Franka Emika Panda | 7 | 토크 센서 내장, 임피던스 제어 |
| **Follower Arm** (수행) | UR5e | 6 | 5kg 페이로드, 850mm 도달거리 |
| **대안** | Franka Emika Panda × 2 | 7+7 | 동일 아키텍처로 Leader/Follower 통일 |

> **시뮬레이션 전용**: 실제 로봇이 없어도 Isaac Sim 내에서 두 팔 모두 가상 모델로 사용 가능.

### 1.3 Grippers

| 구성 요소 | 모델 | 비고 |
|-----------|------|------|
| **Leader Gripper** | Franka Hand | 2-finger parallel, 최대 개방 80mm |
| **Follower Gripper** | Robotiq 2F-85 | 2-finger parallel, 최대 개방 85mm |
| **Sim Gripper** | Isaac Sim Parallel Gripper | 물리 시뮬레이션 최적화 |

### 1.4 Cameras (팔당 1대, 총 2대)

| 사양 | 값 |
|------|-----|
| 모델 | Intel RealSense D435 |
| 해상도 | 640×480 (RGB), 640×480 (Depth) |
| 프레임레이트 | 30 FPS (최대 90 FPS) |
| 인터페이스 | USB 3.0 |
| 마운트 | 각 Arm의 플랜지에 장착 |

### 1.5 Edge Device (옵션 — Digital Twin 배포용)

| 모델 | GPU | AI 성능 | 비고 |
|------|-----|---------|------|
| Jetson Orin Nano 8GB | 1024-core Ampere | 40 TOPS | 소형, 저전력 |
| Jetson Orin NX 16GB | 1024-core Ampere | 100 TOPS | 적정 성능 |
| **Jetson AGX Orin 64GB** | 2048-core Ampere | **275 TOPS** | **권장** |

### 1.6 Network

| 연결 | 프로토콜 | 대역폭 | 지연 |
|------|----------|--------|------|
| PC ↔ Leader Arm | Ethernet | 1Gbps | < 1ms |
| PC ↔ Follower Arm | Ethernet | 1Gbps | < 1ms |
| PC ↔ Jetson (옵션) | Ethernet | 1Gbps | < 5ms |
| Leader ↔ Follower | ROS2 DDS | 1Gbps | < 10ms |

> **실시간 제어 요구사항**: ROS2 DDS 통신 지연은 10ms 미만이어야 함.
> MoveIt2 Inverse Kinematics 업데이트는 100Hz 이상 권장.

### 1.7 GPU VRAM 요구사항 (VLA Transformer)

| Batch Size | PyTorch FP32 | TensorRT FP16 | TensorRT INT8 |
|------------|-------------|---------------|---------------|
| 1 (추론) | 4.2 GB | 2.1 GB | 1.4 GB |
| 64 | 12.8 GB | 6.4 GB | 4.3 GB |
| 128 | 21.6 GB | 10.8 GB | 7.2 GB |
| 256 | **37.2 GB** | **18.6 GB** | **12.4 GB** |

> RTX 5090 24GB로 FP16 batch_size=256 또는 INT8 batch_size=512까지 가능.

### 1.8 스토리지 추정

| 데이터 유형 | 에피소드 당 크기 | 1,000 에피소드 | 10,000 에피소드 |
|------------|----------------|---------------|----------------|
| RGB-D 영상 (30fps, 5초) | ~50 MB | 50 GB | 500 GB |
| Joint trajectory | ~10 KB | 10 MB | 100 MB |
| Language annotation | ~1 KB | 1 MB | 10 MB |
| VLA checkpoint | - | 2 GB × 10 = 20 GB | 20 GB |
| TensorRT engine | - | 1 GB × 5 = 5 GB | 5 GB |
| **Total** | **~50 MB** | **~75 GB** | **~525 GB** |

---

## 2. 소프트웨어 요구사항

### 2.1 시스템 소프트웨어

| 소프트웨어 | 버전 | 역할 |
|-----------|------|------|
| **Ubuntu** | 22.04 LTS (Jammy) | 호스트 OS |
| **NVIDIA Driver** | 570+ (RTX 5090) / 545+ (RTX 4090) | GPU 드라이버 |
| **CUDA** | 12.6+ | GPU 컴퓨팅 |
| **Docker** | 27.0+ | Isaac Sim 컨테이너 |
| **NVIDIA Container Toolkit** | 1.16+ | Docker GPU passthrough |
| **Python** | 3.10 | 모든 Conda 환경 |

### 2.2 NVIDIA 스택

| 구성 요소 | 버전 | 설치 방식 | 역할 |
|-----------|------|----------|------|
| **Isaac Sim** | **2025.2.0** | Docker (`nvcr.io/nvidia/isaac-sim:2025.2.0`) | 로봇 시뮬레이션 |
| **Isaac Lab** | **2.1** | Conda + pip | VLA 학습 프레임워크 |
| **Cosmos-Reason** | **2.0** | Conda + pip | VLM 태스크 플래너 |
| **Cosmos-Policy** | **2.0** | Conda + pip | VLA 사전 학습 모델 |
| **Cosmos-Transfer** | **2.0** | Conda + pip | Sim-to-Real 이미지 변환 |
| **PhysX** | 5.4 | Isaac Sim 내장 | 물리 엔진 |

### 2.3 ROS2 & 로봇 미들웨어

| 구성 요소 | 버전 | 역할 |
|-----------|------|------|
| **ROS2** | Humble Hawksbill | 분산 로봇 통신 |
| **MoveIt2** | Humble | 모션 플래닝 + IK + 충돌 회피 |
| **ros2_control** | Humble | 하드웨어 인터페이스 |
| **Franka ROS2** | foxy-devel (franka_ros2) | Franka Panda 드라이버 |
| **UR ROS2 Driver** | humble (ur_robot_driver) | UR5e 드라이버 |
| **Intel RealSense ROS2** | humble (realsense2_camera) | RGB-D 카메라 드라이버 |
| **nvblox** | 3.0 (Jetson 전용) | 실시간 3D 재구성 |

### 2.4 딥러닝 프레임워크

| 구성 요소 | 버전 | 역할 |
|-----------|------|------|
| **PyTorch** | 2.5+ | VLA Transformer 학습 |
| **TensorRT** | 10.0+ | 추론 최적화 |
| **ONNX** | 1.16+ | 모델 교환 포맷 |
| **Hugging Face Transformers** | 4.44+ | Vision Transformer 백본 |
| **Flash Attention** | 2.6+ | VLA Transformer 가속 |
| **bitsandbytes** | 0.43+ | 8-bit/4-bit 양자화 |

### 2.5 호환성 매트릭스

```
NVIDIA Driver  ──> CUDA 12.6 ──> PyTorch 2.5+
    │                            │
    ├──> Docker Container Toolkit
    │         │
    │         ├──> Isaac Sim 2025.2  ──> Isaac Lab 2.1
    │         │                              │
    │         │                              ├──> VLA Training
    │         │                              └──> Evaluation
    │         │
    │         └──> Cosmos 2.0 (별도 Conda)
    │                    │
    │                    ├──> Cosmos-Reason (VLM)
    │                    ├──> Cosmos-Policy (VLA)
    │                    └──> Cosmos-Transfer (Sim2Real)
    │
    └──> TensorRT 10.0 ──> ONNX ──> Jetson Inference
```

---

## 3. 시스템 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Development PC                              │
│                                                                     │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │
│  │  Docker:         │  │  Conda:          │  │  Conda:          │  │
│  │  Isaac Sim       │  │  Isaac Lab       │  │  Cosmos 2.0      │  │
│  │  2025.2          │  │  2.1             │  │  (Reason/Policy) │  │
│  │                  │  │                  │  │                  │  │
│  │  양팔 시뮬레이션  │  │  VLA RL 학습     │  │  VLM 태스크 플랜  │  │
│  │  CosmosWriter    │  │  BC + PPO       │  │  이미지 생성     │  │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘  │
│           │                    │                     │           │
└───────────┼────────────────────┼─────────────────────┼───────────┘
            │                    │                     │
            │    ROS2 DDS (Ethernet / Shared Memory)    │
            │                    │                     │
┌───────────┼────────────────────┼─────────────────────┼───────────┐
│           ▼                    ▼                     ▼           │
│  ┌───────────────────────────────────────────────────────────┐   │
│  │                    Digital Twin Loop                       │   │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐       │   │
│  │  │Data  │─►│ Gap  │─►│ Auto │─►│Policy│─►│Orch  │       │   │
│  │  │Logger│  │Anal. │  │Retrain│  │Reg.  │  │est.  │       │   │
│  │  └──────┘  └──────┘  └──────┘  └──────┘  └──────┘       │   │
│  └───────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────┐          ┌─────────────────────┐         │
│  │  Leader Arm         │          │  Follower Arm       │         │
│  │  (Franka Panda +    │◄───ETHER►│  (UR5e + Jetson)    │         │
│  │   RealSense D435)   │          │   RealSense D435)   │         │
│  └─────────────────────┘          └─────────────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. 저장소 구조

```
Leader_Follower_arm_ROS_VLA/
├── docs/           # 문서 (11개)
├── src/            # 소스 코드
│   ├── leader_arm/       # Leader Arm 시연/녹화
│   ├── cosmos_reason/    # VLM 태스크 플래너
│   ├── vla_policy/       # VLA Transformer
│   ├── follower_arm/     # Follower Arm 제어
│   ├── isaac_sim/        # Isaac Sim 환경
│   ├── isaac_lab/        # Isaac Lab 학습
│   └── digital_twin/     # Digital Twin (5모듈)
├── config/         # 설정 파일
├── docker/         # Dockerfile
├── scripts/        # 셸 스크립트
├── urdf/           # 로봇 모델
├── data/           # 에피소드 데이터 (gitignore)
└── results/        # 실험 결과 (gitignore)
```

---

## 5. 네트워크 포트

| 포트 | 용도 | 프로토콜 |
|------|------|----------|
| 9090 | Isaac Sim WebRTC | TCP |
| 9100 | Isaac Sim Omniverse | TCP |
| 11311 | ROS2 Discovery Server | UDP |
| 7400-7500 | ROS2 DDS (RTPS) | UDP |
| 50051 | Cosmos Reason gRPC | TCP |
| 50052 | Cosmos Policy gRPC | TCP |
