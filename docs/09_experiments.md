# Experiment Scenarios & Test Plans

> 검증된 실험 시나리오 7개 — 단일 Pick-and-Place에서 양팔 협업까지

---

## Experiment 1: Single Arm Pick-and-Place (기본)

### Objective
Follower Arm(VLA 정책)이 Leader Arm 시연을 모방하여 Pick-and-Place 수행

### Setup
| 항목 | 값 |
|------|-----|
| Leader Arm | Franka Panda (시뮬레이션) |
| Follower Arm | UR5e (시뮬레이션) |
| Objects | Blue block (3cm³), Red box (5cm³) |
| Camera | Leader RGB-D @ 640×480, 30Hz |
| Language | "Pick the blue block and place it on the red box" |

### Procedure
1. **Data Collection**: Leader Arm pick-and-place 50회 시연 (변형된 초기 조건)
2. **Training**: BC (50 epochs) → PPO (5000 iterations)
3. **Sim Evaluation**: 100 trials (랜덤화된 초기 위치)
4. **Real Evaluation**: 50 trials (동일 환경)

### Metrics

| 메트릭 | 계산식 | 목표 |
|--------|--------|------|
| Task Success Rate | 성공 / 전체 태스크 | > 90% (sim), > 80% (real) |
| Cycle Time | 태스크 완료 시간 | < 10초 |
| Trajectory Smoothness | jerk RMS | < 500 rad/s³ |
| Grasp Success Rate | 성공 그립 / 전체 시도 | > 95% |
| Collision Rate | 충돌 태스크 / 전체 | < 5% |

### Expected Results
- BC only: ~60-70% SR (sim)
- BC + PPO: ~85-95% SR (sim)
- Sim-to-Real: ~70-85% SR (real)
- After Digital Twin fine-tune: ~85-90% SR (real)

### Troubleshooting

| 문제 | 원인 | 해결 |
|------|------|------|
| 그립 실패 | 잘못된 그립 자세 | 시연 데이터 증강 (다양한 접근 각도) |
| 물체 놓침 | 그리퍼 힘 부족 | 그리퍼 제어 gain 조정 |
| 충돌 발생 | MoveIt2 collision margin 부족 | collision_padding 증가 (0.01 → 0.03) |
| 궤적 지터 | VLA action noise | temporal smoothing (0.5 EMA) |

---

## Experiment 2: Language-Conditioned Manipulation

### Objective
Cosmos Reason VLM으로 자연어 명령 → 태스크 플랜 → VLA 실행

### Setup
| 항목 | 값 |
|------|-----|
| Leader Arm | Franka Panda |
| Follower Arm | UR5e |
| Instruction Set | 20개 다양한 자연어 명령 |
| Objects | 5종 색상/형태 블록 |

### Instruction Examples
```
"Pick the red cube and place it on the green box"
"Move the blue cylinder to the left side of the table"
"Stack the yellow block on top of the red block"
"Push the green triangle toward the center"
"Grasp the object and lift it 20cm"
```

### Procedure
1. 100회 시연 (각기 다른 언어 명령)
2. Cosmos Reason 태스크 플랜 검증 (정확도 측정)
3. VLA 학습 (언어 임베딩 조건)
4. 20개 명령어 × 5회 = 100회 평가

### Metrics

| 메트릭 | 목표 |
|--------|------|
| Language Compliance Rate | > 90% |
| Task Plan Accuracy | > 95% (VLM 평가) |
| Task Success Rate | > 80% |
| Instruction Following Latency | < 500ms (VLM 추론) |

---

## Experiment 3: Object Generalization

### Objective
학습하지 않은 새로운 물체에 대한 VLA 일반화 능력 측정

### Setup
| 항목 | Train Set | Test Set |
|------|-----------|----------|
| Objects | Blue cube, Red box, Green cylinder | Yellow sphere, Purple pyramid, Orange cone |
| Shape | 3종 (cube, box, cylinder) | 3종 (sphere, pyramid, cone) |
| Color | 3종 | 3종 |
| Size | 3-5cm | 3-5cm |

### Procedure
1. 5개 물체로 100회 시연 학습
2. 3개 새로운 물체로 Zero-shot 평가 (50회)
3. 10회 시연 후 Fine-tuning 평가 (50회)

### Metrics

| 메트릭 | Zero-shot | 10-shot |
|--------|-----------|---------|
| Grasp Success | > 50% | > 80% |
| Task Success | > 40% | > 75% |
| Visual Feature Alignment | > 0.7 cosine sim | — |

### Analysis
VLA의 vision encoder가 shape, color, size 중 어떤 feature에 민감한지 분석:
```
Generalization breakdown:
├── Shape novel:  72%  ← shape generalization
├── Color novel:  85%  ← color generalization  
├── Size novel:   78%  ← scale generalization
└── All novel:    45%  ← complete novel
```

---

## Experiment 4: Scene Perturbation Robustness

### Objective
환경 변화에 대한 VLA 정책 강건성 평가

### Perturbation Types

| 유형 | 변수 | 범위 | 단계 |
|------|------|------|------|
| Lighting | Intensity | 100 ~ 800 lux | 5단계 |
| Lighting | Direction | -60° ~ +60° | 5단계 |
| Clutter | Extra objects | 0 ~ 5개 | 3단계 |
| Occlusion | Object遮挡 | 0 ~ 75% | 4단계 |
| Camera Noise | Gaussian σ | 0 ~ 0.05 | 4단계 |
| Background | Texture | 5종 패턴 | 5종 |

### Procedure
1. Baseline: 표준 조건 50회
2. 각 perturbation 유형별 50회 (총 300회)
3. Grad-CAM으로 attention 변화 시각화

### Results Template

```
Perturbation Robustness Report
================================
Baseline SR: 92%

Lighting (200 lux):     88% (-4%)
Lighting (600 lux):     85% (-7%)
Clutter (3 objects):    82% (-10%)
Occlusion (50%):        70% (-22%)  ← 가장 취약
Camera Noise (0.03):    86% (-6%)

→ Recommendation: occlusion augmentation 추가
```

---

## Experiment 5: Bimanual Coordination

### Objective
Leader-Follower 양팔 협업 태스크

### Task: "Hold container, insert object"

```
Leader Arm: 용기 잡기 (hold position)
Follower Arm: 물체 집어서 용기에 넣기 (insert)
```

### Setup
| 항목 | 값 |
|------|-----|
| Leader Task | 용기 잡기 (정적 pose 유지) |
| Follower Task | 물체 pick → 용기에 insert |
| Leader Stiffness | 높음 (임피던스 제어) |
| Follower Compliance | 중간 (VLA 제어) |

### Procedure
1. 100회 bimanual 시연 (Leader: teleop, Follower: recorded)
2. Centralized VLA 학습 (joint action space)
3. 100회 평가 (50회 sim + 50회 real)

### Metrics

| 메트릭 | 목표 |
|--------|------|
| Bimanual Task SR | > 80% |
| Coordination Delay | < 200ms |
| Inter-arm Min Distance | > 5cm |
| Collision Rate | < 3% |
| Insert Success Rate | > 85% |

---

## Experiment 6: Long-Horizon Tasks

### Objective
다단계 복합 태스크 수행 능력 평가

### Task: Multi-step assembly

```
Step 1: "Pick the blue block"        → grasp
Step 2: "Place it on the base"       → place
Step 3: "Pick the red cap"           → regrasp
Step 4: "Place cap on blue block"    → stack
Step 5: "Push the assembly to center" → push
```

### Hierarchical VLA 구조

```
┌─────────────────────────────────────────────────────────────┐
│              Hierarchical VLA                                │
│                                                              │
│  ┌──────────────────────┐  ┌──────────────────────────────┐ │
│  │  High-Level Policy   │  │  Low-Level VLA Policy        │ │
│  │  (task selection)    │  │  (motion generation)         │ │
│  │                      │  │                              │ │
│  │  Input: scene + goal │  │  Input: subgoal + state      │ │
│  │  Output: subgoal     │──►│  Output: joint velocities   │ │
│  └──────────────────────┘  └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Metrics

| 메트릭 | 목표 |
|--------|------|
| Cumulative Success Rate | > 60% (4/5 steps) |
| Step Completion Rate | > 85% per step |
| Recovery After Failure | > 50% |
| Average Steps per Task | < 6 (retry 포함) |

---

## Experiment 7: Sim-to-Real Transfer

### Objective
시뮬레이션 전용 학습 → 실제 로봇 Zero-shot 전이 + Fine-tuning 효율

### Setup
| Phase | Training Data | Environment |
|-------|--------------|-------------|
| Pretrain | 500 episodes (sim) | Isaac Sim with domain randomization |
| Zero-shot | 0 episodes | Real world |
| Fine-tune 1 | 10 episodes (real) | Real world |
| Fine-tune 2 | 50 episodes (real) | Real world |

### Domain Randomization for Sim2Real

```python
sim2real_dr_config = {
    "lighting": {"intensity": (300, 800)},
    "camera_noise": {"sigma": (0.001, 0.03)},
    "friction": {"table": (0.2, 1.0)},
    "joint_damping": {"leader": (0.01, 0.2), "follower": (0.01, 0.2)},
    "payload": {"mass": (0.0, 0.5)},
    "lens_distortion": {"coefficient": (-0.01, 0.01)},
    "background": {"textures": ["lab", "warehouse", "home"]},
}
```

### Metrics

| Phase | Sim Success Rate | Real Success Rate | Gap |
|-------|-----------------|-------------------|-----|
| Pretrain (Sim) | 92% | — | — |
| Zero-shot | — | 55% | **37%** |
| + 10 real demos | 93% | 75% | **18%** |
| + 50 real demos | 94% | 88% | **6%** |

### Digital Twin Integration

```
Sim SR: 92% vs Real SR: 55% → Gap: 37% (> 15% threshold)
    → Digital Twin triggers retrain
    → Cosmos-Transfer로 real 데이터 증강
    → 10 real demo fine-tune
    → Real SR: 55% → 75% ✅
```

---

## Appendix: Experiment Execution Script

```python
# scripts/run_experiment.py
"""
실험 실행 프레임워크
"""
import argparse
import json
import time
from datetime import datetime


class ExperimentRunner:
    """
    실험 관리자
    - 실험 설정 로드
    - 시뮬레이션/실제 환경에서 실행
    - 메트릭 수집 및 저장
    """
    
    def __init__(self, experiment_id: str, config_path: str):
        self.experiment_id = experiment_id
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.results = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "config": self.config,
            "trials": [],
            "metrics": {},
        }
    
    def run_trial(self, trial_id: int, params: dict) -> dict:
        """단일 시험 실행"""
        start_time = time.time()
        
        # 실행 (실제 환경 호출)
        success = self._execute_task(params)
        
        end_time = time.time()
        
        return {
            "trial_id": trial_id,
            "success": success,
            "duration": end_time - start_time,
            "params": params,
        }
    
    def run_experiment(self, num_trials: int):
        """전체 실험 실행"""
        for i in range(num_trials):
            params = self._generate_trial_params(i)
            trial_result = self.run_trial(i, params)
            self.results["trials"].append(trial_result)
            
            # 진행률 출력
            if (i + 1) % 10 == 0:
                sr = sum(t["success"] for t in self.results["trials"]) / (i + 1)
                print(f"  Trial {i+1}/{num_trials} | Current SR: {sr:.1%}")
        
        # 메트릭 계산
        self._compute_metrics()
        
        # 저장
        output_path = f"results/{self.experiment_id}.json"
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"Experiment complete: {output_path}")
        return self.results
    
    def _execute_task(self, params) -> bool:
        """실제 태스크 실행 (상속하여 구현)"""
        raise NotImplementedError
    
    def _generate_trial_params(self, trial_id: int) -> dict:
        """시험 파라미터 생성"""
        return {"trial_id": trial_id}
    
    def _compute_metrics(self):
        """메트릭 계산"""
        trials = self.results["trials"]
        successes = [t["success"] for t in trials]
        
        self.results["metrics"] = {
            "success_rate": sum(successes) / len(successes),
            "total_trials": len(trials),
            "successful_trials": sum(successes),
            "avg_duration": sum(t["duration"] for t in trials) / len(trials),
        }
```
