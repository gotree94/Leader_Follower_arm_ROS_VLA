# Leader_Follower_arm_ROS_VLA

> **NVIDIA Isaac Sim + Cosmos VLA 기반 Leader-Follower 로봇팔 시스템**
> 시연(demonstration) → VLA 학습 → 자율 수행 → **디지털 트윈 자동 개선**

---

## 프로젝트 개요

**Leader-Follower Arm VLA**는 NVIDIA의 최신 AI 기술 스택을 활용하여 로봇팔이 사람의 시연과 자연어 명령을 통해 스스로 학습하고, 디지털 트윈을 통해 지속적으로 성능을 개선하는 **완전 자동화 파이프라인**입니다.

### 핵심 시나리오

```
[사람] ─── "파란 블록 집어서 빨간 상자에 넣어줘"
    │
    ├─── ① Leader Arm이 시연 (조이스틱으로 데모)
    ├─── ② Cosmos Reason이 언어 명령 이해 → 태스크 플랜
    ├─── ③ Isaac Sim에서 학습 데이터 생성
    ├─── ④ Isaac Lab VLA Transformer 학습 (BC → PPO)
    ├─── ⑤ Follower Arm이 자율 수행
    └─── ⑥ Digital Twin이 지속적 개선
```

### 차별점

| 구분 | 기존 로봇팔 | 이 프로젝트 |
|------|-----------|-----------|
| **프로그래밍** | 수동 조인트 각도 지정 | 자연어 명령 + 시연 |
| **학습** | 없음 (고정 동작) | VLA Transformer (BC + PPO) |
| **일반화** | 불가능 | 새로운 물체/환경에 적응 |
| **개선** | 수동 튜닝 | Digital Twin 자동 재학습 |
| **확장성** | 단일 태스크 | 언어로 새로운 태스크 추가 |

---

## 시스템 요구사항

### 필수 하드웨어

| 구성 요소 | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | RTX 4090 24GB | **RTX 5090 24GB** |
| **RAM** | 32GB | **64GB** |
| **CPU** | 8코어 | 16코어 |
| **Storage** | NVMe 500GB | NVMe 2TB+ |
| **OS** | Ubuntu 22.04 LTS | Ubuntu 22.04 LTS |

### 로봇 하드웨어

| 역할 | 로봇 | DOF | 비고 |
|------|------|-----|------|
| **Leader Arm** | Franka Emika Panda | 7 | 시연 담당 (또는 Isaac Sim 가상) |
| **Follower Arm** | UR5e | 6 | VLA 정책 실행 (또는 Isaac Sim 가상) |
| **Gripper** | Franka Hand / Robotiq 2F-85 | 1 | 병렬 그리퍼 |
| **Camera** | Intel RealSense D435 × 2 | - | RGB-D @ 640×480, 30fps |
| **Edge (옵션)** | Jetson Orin NX/AGX | - | 실시간 VLA 추론 |

### 소프트웨어 스택

| 구성 요소 | 버전 | 역할 |
|-----------|------|------|
| **Isaac Sim** | 2025.2.0 | 양팔 시뮬레이션, 합성 데이터 |
| **Isaac Lab** | 2.1 | VLA Transformer 학습 |
| **Cosmos Reason** | 2.0 | VLM 언어 명령 이해 |
| **Cosmos Policy** | 2.0 | VLA 사전 학습 모델 |
| **Cosmos Transfer** | 2.0 | Sim-to-Real 변환 |
| **ROS2** | Humble | 분산 로봇 통신 |
| **MoveIt2** | Humble | IK + 충돌 회피 |
| **TensorRT** | 10.0 | 추론 최적화 |
| **PyTorch** | 2.5+ | VLA 모델 학습 |

---

## 프로젝트 구조

```
Leader_Follower_arm_ROS_VLA/
│
├── README.md                          # 프로젝트 개요
│
├── docs/                              # 문서 (11개)
│   ├── 01_prerequisites.md            # 시스템 요구사항
│   ├── 02_environment_setup.md        # 환경 구성 가이드
│   ├── 03_arm_urdf_modeling.md        # Franka/UR5e URDF
│   ├── 04_isaac_sim_arms.md          # 양팔 시뮬레이션 환경
│   ├── 05_cosmos_reason_vlm.md        # VLM 태스크 플래너
│   ├── 06_vla_training.md            # VLA Transformer 학습
│   ├── 07_vla_inference.md            # 실시간 VLA 추론
│   ├── 08_bimanual_extension.md       # 양팔 협업 확장
│   ├── 09_experiments.md              # 실험 시나리오
│   ├── 10_architecture_overview.md    # 아키텍처 개요 (쉬운 설명)
│   └── 11_digital_twin_loop.md        # 디지털 트윈 파이프라인
│
├── src/                               # 소스 코드
│   ├── leader_arm/                    # Leader Arm 시연
│   │   ├── teleop_joy.py              #   조이스틱 원격 조작
│   │   ├── trajectory_recorder.py     #   궤적 녹화
│   │   └── leader_driver.py           #   하드웨어 드라이버
│   │
│   ├── cosmos_reason/                 # VLM 언어 이해
│   │   ├── task_planner.py            #   자연어 → 태스크 플랜
│   │   ├── object_detector.py         #   객체 인식
│   │   └── scene_graph.py             #   장면 그래프
│   │
│   ├── vla_policy/                    # VLA 정책 (핵심)
│   │   ├── vla_network.py             #   Transformer VLA
│   │   ├── export_onnx.py             #   ONNX 변환
│   │   ├── cosmos_policy_finetune.py  #   Cosmos-Policy LoRA
│   │   └── vla_ros_node.py            #   ROS2 추론 노드
│   │
│   ├── follower_arm/                  # Follower Arm 제어
│   │   ├── arm_controller.py          #   MoveIt2 + VLA
│   │   ├── gripper_controller.py      #   그리퍼 제어
│   │   └── safety_monitor.py          #   충돌 감지
│   │
│   ├── isaac_sim/                     # 시뮬레이션
│   │   ├── setup_bimanual_scene.py    #   양팔 환경
│   │   ├── cosmos_sdg_pipeline.py     #   CosmosWriter SDG
│   │   └── domain_randomize.py        #   도메인 랜덤화
│   │
│   ├── isaac_lab/                     # VLA 학습
│   │   ├── arm_vla_cfg.py             #   환경 설정
│   │   ├── arm_vla_task.py            #   태스크 정의
│   │   └── train_vla.py               #   학습 실행
│   │
│   └── digital_twin/                  # 디지털 트윈 (4단계)
│       ├── data_logger.py             #   Phase 7: 데이터 수집
│       ├── gap_analyzer.py            #   Phase 8: 갭 분석
│       ├── auto_retrain_pipeline.py   #   Phase 9: 자동 재학습
│       ├── policy_registry.py         #   정책 버전 관리
│       └── orchestrator.py            #   Phase 10: 오케스트레이터
│
├── config/                            # 설정 파일
│   ├── moveit_params.yaml             # MoveIt2 설정
│   ├── cosmos_config.yaml             # Cosmos 설정
│   └── digital_twin_config.yaml       # Digital Twin 설정
│
├── docker/                            # Docker
│   ├── Dockerfile.isaac               # Isaac Sim Dockerfile
│   └── docker-compose.yaml            # Compose 구성
│
├── scripts/                           # 실행 스크립트
│   ├── setup_host.sh                  # 호스트 설정
│   ├── setup_jetson.sh                # Jetson 설정
│   ├── run_vla_training.sh            # VLA 학습 파이프라인
│   └── deploy_policy.sh               # Blue-Green 배포
│
├── urdf/                              # 로봇 모델
├── data/                              # 에피소드 데이터
└── results/                           # 실험 결과
```

---

## 핵심 기술

### 1. VLA (Vision-Language-Action) Transformer

```
입력: RGB-D 이미지 + 조인트 각도 + 자연어 명령
  │
  ├── Vision Encoder: ResNet-50 → 이미지 특징
  ├── Language Encoder: Cosmos Reason 임베딩
  └── Proprioception: 조인트 + 그리퍼 MLP
  │
  Cross-Attention Transformer (4 layers, 8 heads)
  │
출력: 조인트 속도 (6-DOF) + 그리퍼 명령
```

### 2. Cosmos Reason (VLM)

자연어 명령을 로봇이 이해하는 태스크 플랜으로 변환:
```
"파란 블록 집어서 빨간 상자에 넣어줘"
  → [approach, grasp, lift, move, place, release]
  → 각 단계별 목표 위치 및 그리퍼 상태 포함
```

### 3. Digital Twin Closed-Loop

| Phase | 구성 요소 | 기능 |
|-------|----------|------|
| **Phase 7** | Data Logger | 실제 로봇 궤적 + 영상 DB 저장 |
| **Phase 8** | Gap Analyzer | Sim-vs-Real 성능 차이 분석 (임계값: 15%) |
| **Phase 9** | Auto Retrain | 실패 데이터 → LoRA fine-tuning → TensorRT |
| **Phase 10** | Orchestrator | 5분 주기 모니터링 → 자동 트리거 |

---

## 설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론
git clone <this-repo>
cd Leader_Follower_arm_ROS_VLA

# 호스트 설정 스크립트 실행
bash scripts/setup_host.sh
```

### 2. Isaac Sim 실행

```bash
bash ~/run_isaac_sim.sh
```

### 3. 학습 실행

```bash
bash scripts/run_vla_training.sh
```

### 4. Digital Twin Orchestrator 실행

```bash
conda activate isaaclab
python src/digital_twin/orchestrator.py --start
```

---

## 실험 시나리오

| # | 실험 | 핵심 메트릭 | 목표 |
|---|------|-----------|------|
| 1 | Pick-and-Place 기본 | Task Success Rate | > 90% (sim) / > 80% (real) |
| 2 | 언어 조건부 조작 | Language Compliance | > 90% |
| 3 | 객체 일반화 | Zero-shot SR | > 50% |
| 4 | 환경 교란 강건성 | Perturbation SR | > 80% |
| 5 | 양팔 협업 | Bimanual SR | > 80% |
| 6 | 장기 태스크 | Cumulative SR | > 60% (4/5 단계) |
| 7 | Sim-to-Real 전이 | Sim2Real Gap | Gap < 15% |

---

## 기술 스택

```
Application Layer: ROS2 Humble (DDS)
    │
    ▼
AI Layer: Cosmos 2.0 (Reason + Policy + Transfer) + Isaac Lab 2.1
    │
    ▼
Simulation Layer: Isaac Sim 2025.2 + PhysX 5 + TensorRT 10
    │
    ▼
Hardware Layer: Dev PC (RTX 5090) + Franka Panda + UR5e + Jetson Orin
```

---

## 문서 로드맵

| 문서 | 내용 | 필수 |
|------|------|------|
| docs/01_prerequisites.md | 하드웨어/소프트웨어 요구사항 | ✅ |
| docs/02_environment_setup.md | Docker, Conda, ROS2, MoveIt2 설치 | ✅ |
| docs/03_arm_urdf_modeling.md | Franka/UR5e URDF, USD 변환 | ✅ |
| docs/04_isaac_sim_arms.md | 양팔 시뮬레이션, CosmosWriter SDG | ✅ |
| docs/05_cosmos_reason_vlm.md | VLM 태스크 플래너, 객체 탐지 | ✅ |
| docs/06_vla_training.md | VLA Transformer BC + PPO 학습 | ✅ |
| docs/07_vla_inference.md | 실시간 추론, TensorRT, MoveIt2 | ✅ |
| docs/08_bimanual_extension.md | 양팔 협업, 충돌 회피 | ✅ |
| docs/09_experiments.md | 7개 실험 시나리오 | ✅ |
| docs/10_architecture_overview.md | **쉬운 설명** (첫 번째로 읽을 것) | ✅ |
| docs/11_digital_twin_loop.md | 4단계 디지털 트윈 파이프라인 | ✅ |

---

## 라이선스

이 프로젝트는 NVIDIA Isaac Sim, Cosmos, Isaac Lab을 기반으로 합니다.
각 구성 요소의 라이선스를 별도로 확인하세요.
