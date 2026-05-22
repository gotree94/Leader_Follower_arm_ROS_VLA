# Digital Twin Closed-Loop for Arm VLA

> 데이터 수집 → 갭 분석 → 자동 재학습 → 무중단 배포의 완전 순환 파이프라인

---

## 1. Digital Twin 개요

TurtleBot3 프로젝트의 Digital Twin 구조를 Arm VLA에 특화하여 적용합니다.

### 1.1 왜 Arm VLA에 Digital Twin이 필요한가?

로봇팔은 TurtleBot3보다 변수가 많습니다:

| 변수 | TurtleBot3 | Arm VLA | 영향 |
|------|-----------|---------|------|
| 조인트 | 2개 (바퀴) | **13개** (팔 2대) | 기하급수적 상태 공간 |
| 액션 | 속도 + 각도 | **속도 + 그립 + 힘** | 더 복잡한 정책 |
| 환경 | 바닥 + 벽 | **테이블 + 물체 + 장애물** | 더 다양한 상호작용 |
| 센서 | LiDAR + IMU | **RGB-D × 2 + 조인트** | 고차원 관측 |
| 언어 | ❌ | **자연어 명령** | 추상적 조건 |

→ 시뮬레이션과 실제의 차이가 더 크고 빈번 → **Digital Twin이 필수적**

### 1.2 순환 구조

```
                    ┌─────────────────────────────────────────────┐
                    │         Digital Twin Loop (Arm VLA)          │
                    │                                              │
                    │  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
  [실제 로봇] ──────►│  Phase 7  │─►│  Phase 8 │─►│  Phase 9 │──┼──► [새 정책 배포]
  [실제 환경]        │  Data     │  │  Gap     │  │  Auto    │  │
                    │  Logger   │  │  Analyzer│  │  Retrain │  │
                    └──────────┘  └──────────┘  └──────────┘  │
                         │              │              │        │
                         ▼              ▼              ▼        │
                    ┌──────────┐  ┌──────────┐  ┌──────────┐  │
                    │  Episode │  │  Metrics │  │  Policy  │  │
                    │  DB      │  │  Dashboard│  │ Registry │  │
                    └──────────┘  └──────────┘  └──────────┘  │
                         │                                      │
                         └────────── Phase 10: Orchestrator ────┘
                                              │
                                          [5분 주기 모니터링]
```

---

## 2. Phase 7: Data Logger (Arm VLA)

실제 Follower Arm의 모든 동작 데이터를 수집하여 Episode DB에 저장합니다.

### 2.1 수집 데이터

| 데이터 유형 | 설명 | 크기/에피소드 |
|-----------|------|-------------|
| **RGB 영상** | Leader view + Follower view (640×480, 30fps) | ~30 MB |
| **Depth 영상** | 각 view별 depth map | ~10 MB |
| **Joint Trajectory** | 13개 조인트 각도 + 속도 (100Hz) | ~50 KB |
| **Gripper State** | 2개 그리퍼 개폐 정도 | ~1 KB |
| **EE Pose** | End-effector 위치/회전 (7-DOF) | ~5 KB |
| **Language Instruction** | 원본 자연어 명령 | ~1 KB |
| **Task Plan** | Cosmos Reason의 태스크 플랜 | ~2 KB |
| **VLA Action** | VLA 정책 출력 (추론 결과) | ~5 KB |
| **Reward/Success** | 태스크 성공/실패 + 단계별 보상 | ~1 KB |
| **Force/Torque** | 손목 힘/토크 센서 (옵션) | ~10 KB |
| ****Total**** | | **~50 MB/에피소드** |

### 2.2 DB 스키마

```sql
-- Episode DB (SQLite)
CREATE TABLE episodes (
    episode_id TEXT PRIMARY KEY,
    timestamp TEXT NOT NULL,
    arm_type TEXT NOT NULL,  -- 'leader', 'follower', 'bimanual'
    task_type TEXT NOT NULL,
    instruction TEXT,
    success BOOLEAN DEFAULT 0,
    duration_ms INTEGER,
    num_frames INTEGER,
    sim_origin BOOLEAN DEFAULT 0,  -- True = from Isaac Sim
    model_version TEXT,  -- VLA policy version used
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE frames (
    frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
    episode_id TEXT NOT NULL,
    timestep INTEGER NOT NULL,
    timestamp REAL NOT NULL,
    
    -- Joint states (JSON arrays)
    leader_joint_positions TEXT,    -- 7 floats
    leader_joint_velocities TEXT,   -- 7 floats
    follower_joint_positions TEXT,   -- 6 floats
    follower_joint_velocities TEXT,  -- 6 floats
    
    -- Gripper
    leader_gripper REAL,
    follower_gripper REAL,
    
    -- EE poses (JSON arrays)
    leader_ee_pose TEXT,     -- [x,y,z,qw,qx,qy,qz]
    follower_ee_pose TEXT,   -- [x,y,z,qw,qx,qy,qz]
    
    -- VLA inference
    vla_action TEXT,         -- [6 joint vel + 1 gripper]
    vla_value REAL,
    vla_success_prob REAL,
    
    -- Rewards
    reward REAL DEFAULT 0.0,
    task_progress REAL DEFAULT 0.0,
    collision BOOLEAN DEFAULT 0,
    
    -- Media file paths
    leader_rgb_path TEXT,
    leader_depth_path TEXT,
    follower_rgb_path TEXT,
    follower_depth_path TEXT,
    
    FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
);

CREATE INDEX idx_frames_episode ON frames(episode_id);
CREATE INDEX idx_episodes_success ON episodes(success);
CREATE INDEX idx_episodes_type ON episodes(task_type);
```

### 2.3 Data Logger 구현

```python
# src/digital_twin/data_logger.py
"""
Arm VLA 실시간 데이터 로거
"""
import sqlite3
import json
import time
import os
import threading
import numpy as np
from datetime import datetime
from queue import Queue
from typing import Optional


class ArmEpisodeLogger:
    """
    Follower Arm 에피소드 데이터 로거
    
    - ROS2 콜백에서 데이터 수신
    - 백그라운드 스레드로 DB에 비동기 저장
    - 자동 버퍼링 (100프레임 or 5초 단위)
    """
    
    def __init__(
        self,
        db_path: str = "data/episode_db.sqlite",
        video_dir: str = "data/episodes",
        buffer_size: int = 100,
        auto_flush_interval: float = 5.0,
    ):
        self.db_path = db_path
        self.video_dir = video_dir
        self.buffer_size = buffer_size
        self.auto_flush_interval = auto_flush_interval
        
        # 버퍼
        self.buffer = Queue()
        self.current_episode = None
        self.frame_count = 0
        
        # DB 초기화
        self._init_db()
        
        # 비디오 저장 디렉토리
        os.makedirs(video_dir, exist_ok=True)
        
        # 백그라운드 플러시 스레드
        self._flush_thread = threading.Thread(target=self._flush_loop, daemon=True)
        self._flush_thread.start()
        
        print(f"ArmEpisodeLogger initialized: {db_path}")
    
    def _init_db(self):
        """DB 초기화 및 테이블 생성"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=OFF")
        
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                timestamp TEXT,
                arm_type TEXT,
                task_type TEXT,
                instruction TEXT,
                success BOOLEAN DEFAULT 0,
                duration_ms INTEGER,
                num_frames INTEGER,
                sim_origin BOOLEAN DEFAULT 0,
                model_version TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS frames (
                frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
                episode_id TEXT,
                timestep INTEGER,
                timestamp REAL,
                leader_joint_positions TEXT,
                leader_joint_velocities TEXT,
                follower_joint_positions TEXT,
                follower_joint_velocities TEXT,
                leader_gripper REAL,
                follower_gripper REAL,
                leader_ee_pose TEXT,
                follower_ee_pose TEXT,
                vla_action TEXT,
                vla_value REAL,
                vla_success_prob REAL,
                reward REAL DEFAULT 0.0,
                task_progress REAL DEFAULT 0.0,
                collision BOOLEAN DEFAULT 0,
                leader_rgb_path TEXT,
                leader_depth_path TEXT,
                follower_rgb_path TEXT,
                follower_depth_path TEXT,
                FOREIGN KEY (episode_id) REFERENCES episodes(episode_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_frames_episode ON frames(episode_id);
            CREATE INDEX IF NOT EXISTS idx_episodes_success ON episodes(success);
        """)
    
    def start_episode(
        self,
        episode_id: str,
        task_type: str = "unknown",
        instruction: str = "",
        arm_type: str = "follower",
        model_version: str = "unknown",
        sim_origin: bool = False,
    ):
        """새 에피소드 시작"""
        self.current_episode = {
            "episode_id": episode_id,
            "timestamp": datetime.now().isoformat(),
            "arm_type": arm_type,
            "task_type": task_type,
            "instruction": instruction,
            "model_version": model_version,
            "sim_origin": sim_origin,
            "start_time": time.time(),
            "frames": [],
        }
        self.frame_count = 0
    
    def log_frame(
        self,
        timestep: int,
        timestamp: float,
        leader_joints: np.ndarray,
        follower_joints: np.ndarray,
        leader_gripper: float,
        follower_gripper: float,
        leader_ee_pose: Optional[np.ndarray] = None,
        follower_ee_pose: Optional[np.ndarray] = None,
        vla_action: Optional[np.ndarray] = None,
        vla_value: Optional[float] = None,
        vla_success_prob: Optional[float] = None,
        reward: float = 0.0,
        task_progress: float = 0.0,
        collision: bool = False,
        leader_rgb: Optional[np.ndarray] = None,
        leader_depth: Optional[np.ndarray] = None,
        follower_rgb: Optional[np.ndarray] = None,
        follower_depth: Optional[np.ndarray] = None,
    ):
        """프레임 데이터 버퍼에 추가"""
        if self.current_episode is None:
            return
        
        frame = {
            "timestep": timestep,
            "timestamp": timestamp,
            "leader_joint_positions": leader_joints[:7].tolist() if leader_joints is not None else None,
            "leader_joint_velocities": None,  # 필요 시 추가
            "follower_joint_positions": follower_joints[:6].tolist() if follower_joints is not None else None,
            "follower_joint_velocities": None,
            "leader_gripper": float(leader_gripper),
            "follower_gripper": float(follower_gripper),
            "leader_ee_pose": leader_ee_pose.tolist() if leader_ee_pose is not None else None,
            "follower_ee_pose": follower_ee_pose.tolist() if follower_ee_pose is not None else None,
            "vla_action": vla_action.tolist() if vla_action is not None else None,
            "vla_value": float(vla_value) if vla_value is not None else None,
            "vla_success_prob": float(vla_success_prob) if vla_success_prob is not None else None,
            "reward": float(reward),
            "task_progress": float(task_progress),
            "collision": int(collision),
        }
        
        # 이미지 저장 (주기적으로)
        if leader_rgb is not None and timestep % 5 == 0:  # 5프레임마다
            frame["leader_rgb_path"] = self._save_image(
                leader_rgb, f"{self.current_episode['episode_id']}/leader_rgb_{timestep:06d}.png"
            )
        if follower_rgb is not None and timestep % 5 == 0:
            frame["follower_rgb_path"] = self._save_image(
                follower_rgb, f"{self.current_episode['episode_id']}/follower_rgb_{timestep:06d}.png"
            )
        
        self.current_episode["frames"].append(frame)
        self.frame_count += 1
        
        # 버퍼 플러시
        if len(self.current_episode["frames"]) >= self.buffer_size:
            self._flush_episode()
    
    def end_episode(self, success: bool):
        """에피소드 종료 및 저장"""
        if self.current_episode is None:
            return
        
        duration = (time.time() - self.current_episode["start_time"]) * 1000
        
        # Episode 메타데이터 저장
        self.conn.execute(
            """INSERT OR REPLACE INTO episodes 
               (episode_id, timestamp, arm_type, task_type, instruction, 
                success, duration_ms, num_frames, model_version, sim_origin)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                self.current_episode["episode_id"],
                self.current_episode["timestamp"],
                self.current_episode["arm_type"],
                self.current_episode["task_type"],
                self.current_episode["instruction"],
                int(success),
                int(duration),
                self.frame_count,
                self.current_episode["model_version"],
                int(self.current_episode["sim_origin"]),
            )
        )
        
        # 프레임 데이터 저장 (배치)
        frames = self.current_episode["frames"]
        self.conn.executemany(
            """INSERT INTO frames 
               (episode_id, timestep, timestamp,
                leader_joint_positions, follower_joint_positions,
                leader_gripper, follower_gripper,
                leader_ee_pose, follower_ee_pose,
                vla_action, vla_value, vla_success_prob,
                reward, task_progress, collision,
                leader_rgb_path, follower_rgb_path)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    self.current_episode["episode_id"],
                    f["timestep"],
                    f["timestamp"],
                    json.dumps(f["leader_joint_positions"]),
                    json.dumps(f["follower_joint_positions"]),
                    f["leader_gripper"],
                    f["follower_gripper"],
                    json.dumps(f["leader_ee_pose"]),
                    json.dumps(f["follower_ee_pose"]),
                    json.dumps(f["vla_action"]),
                    f["vla_value"],
                    f["vla_success_prob"],
                    f["reward"],
                    f["task_progress"],
                    f["collision"],
                    f.get("leader_rgb_path"),
                    f.get("follower_rgb_path"),
                )
                for f in frames
            ]
        )
        
        self.conn.commit()
        print(f"Episode saved: {self.current_episode['episode_id']} ({self.frame_count} frames, {'✓' if success else '✗'})")
        
        self.current_episode = None
        self.frame_count = 0
    
    def _save_image(self, image: np.ndarray, rel_path: str) -> str:
        """이미지 파일 저장"""
        import cv2
        full_path = os.path.join(self.video_dir, rel_path)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        cv2.imwrite(full_path, image)
        return full_path
    
    def _flush_loop(self):
        """주기적 버퍼 플러시"""
        while True:
            time.sleep(self.auto_flush_interval)
            if self.current_episode and len(self.current_episode["frames"]) > 0:
                self._flush_episode()
    
    def _flush_episode(self):
        """현재 버퍼를 DB에 저장"""
        # 백그라운드 플러시 (실제로는 배치 커밋)
        pass
    
    def get_episode_count(self) -> int:
        """저장된 에피소드 수"""
        return self.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
    
    def get_success_rate(self, task_type: Optional[str] = None) -> float:
        """태스크 성공률"""
        if task_type:
            result = self.conn.execute(
                "SELECT AVG(success) FROM episodes WHERE task_type=?", (task_type,)
            ).fetchone()[0]
        else:
            result = self.conn.execute(
                "SELECT AVG(success) FROM episodes"
            ).fetchone()[0]
        return result or 0.0
```

---

## 3. Phase 8: Gap Analyzer (Arm 특화 메트릭)

시뮬레이션 vs 실제 데이터의 차이를 분석하고 재학습 트리거를 결정합니다.

### 3.1 Arm 특화 메트릭

| 메트릭 | 계산식 | 임계값 | 설명 |
|--------|--------|--------|------|
| **Task Success Rate Gap** | Sim_SR - Real_SR | > 15% | 가장 중요한 메트릭 |
| **Grasp Success Rate Gap** | Sim_Grasp - Real_Grasp | > 10% | 그리퍼 특화 |
| **Trajectory Smoothness Gap** | Jerk_RMS(Sim) / Jerk_RMS(Real) | > 2.0 | 궤적 품질 |
| **Cycle Time Gap** | Real_Time / Sim_Time | > 1.3 | 속도 효율 |
| **Collision Frequency Gap** | Real_Collision - Sim_Collision | > 0.05 | 안전성 |
| **Language Compliance Gap** | Sim_Compliance - Real_Compliance | > 10% | 언어 이해 |
| **Force/Contact Gap** | | > 20% | 힘 제어 (옵션) |

### 3.2 Gap Analyzer 구현

```python
# src/digital_twin/gap_analyzer.py
"""
Arm VLA Sim-vs-Real 갭 분석기
"""
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional


@dataclass
class ArmMetrics:
    """Arm 성능 메트릭"""
    task_success_rate: float
    grasp_success_rate: float
    trajectory_smoothness: float  # jerk RMS
    cycle_time_ms: float
    collision_rate: float
    language_compliance: float
    force_error: Optional[float] = None


@dataclass
class GapReport:
    """갭 분석 리포트"""
    timestamp: str
    sim_metrics: ArmMetrics
    real_metrics: ArmMetrics
    gaps: Dict[str, float]
    trigger_retrain: bool
    trigger_reasons: List[str]
    composite_score: float


class ArmGapAnalyzer:
    """
    Arm VLA Sim-vs-Real 갭 분석
    
    시뮬레이션과 실제 Follower Arm의 성능 차이를
    분석하고 재학습 필요성을 판단
    """
    
    def __init__(
        self,
        config_path: str = "config/digital_twin_config.yaml",
    ):
        # 임계값
        self.thresholds = {
            "task_success_rate_gap": 0.15,      # 15%
            "grasp_success_rate_gap": 0.10,      # 10%
            "trajectory_smoothness_gap": 2.0,    # 2x
            "cycle_time_gap": 1.3,               # 1.3x
            "collision_rate_gap": 0.05,          # 5%
            "language_compliance_gap": 0.10,     # 10%
        }
        
        # 가중치 (composite score 계산용)
        self.weights = {
            "task_success_rate_gap": 0.40,
            "grasp_success_rate_gap": 0.15,
            "trajectory_smoothness_gap": 0.10,
            "cycle_time_gap": 0.05,
            "collision_rate_gap": 0.20,
            "language_compliance_gap": 0.10,
        }
        
        # 최소 에피소드 수
        self.min_episodes = 10
    
    def analyze(
        self,
        sim_metrics: ArmMetrics,
        real_metrics: ArmMetrics,
    ) -> GapReport:
        """
        갭 분석 수행
        
        Args:
            sim_metrics: Isaac Sim에서 측정된 메트릭
            real_metrics: 실제 로봇에서 측정된 메트릭
            
        Returns:
            GapReport: 분석 결과
        """
        gaps = {}
        trigger_reasons = []
        trigger_retrain = False
        
        # 1. Task Success Rate Gap
        ts_gap = sim_metrics.task_success_rate - real_metrics.task_success_rate
        gaps["task_success_rate_gap"] = ts_gap
        if ts_gap > self.thresholds["task_success_rate_gap"]:
            trigger_reasons.append(
                f"Task SR gap: {ts_gap:.1%} > {self.thresholds['task_success_rate_gap']:.1%}"
            )
            trigger_retrain = True
        
        # 2. Grasp Success Rate Gap
        gs_gap = sim_metrics.grasp_success_rate - real_metrics.grasp_success_rate
        gaps["grasp_success_rate_gap"] = gs_gap
        if gs_gap > self.thresholds["grasp_success_rate_gap"]:
            trigger_reasons.append(
                f"Grasp SR gap: {gs_gap:.1%} > {self.thresholds['grasp_success_rate_gap']:.1%}"
            )
            trigger_retrain = True
        
        # 3. Trajectory Smoothness Gap
        if real_metrics.trajectory_smoothness > 0:
            sm_gap = sim_metrics.trajectory_smoothness / real_metrics.trajectory_smoothness
        else:
            sm_gap = 1.0
        gaps["trajectory_smoothness_gap"] = sm_gap
        if sm_gap > self.thresholds["trajectory_smoothness_gap"]:
            trigger_reasons.append(
                f"Smoothness gap: {sm_gap:.2f}x > {self.thresholds['trajectory_smoothness_gap']:.1f}x"
            )
            trigger_retrain = True
        
        # 4. Cycle Time Gap
        if real_metrics.cycle_time_ms > 0:
            ct_gap = real_metrics.cycle_time_ms / sim_metrics.cycle_time_ms
        else:
            ct_gap = 1.0
        gaps["cycle_time_gap"] = ct_gap
        if ct_gap > self.thresholds["cycle_time_gap"]:
            trigger_reasons.append(
                f"Cycle time gap: {ct_gap:.2f}x > {self.thresholds['cycle_time_gap']:.1f}x"
            )
            trigger_retrain = True
        
        # 5. Collision Rate Gap
        col_gap = real_metrics.collision_rate - sim_metrics.collision_rate
        gaps["collision_rate_gap"] = col_gap
        if col_gap > self.thresholds["collision_rate_gap"]:
            trigger_reasons.append(
                f"Collision rate gap: {col_gap:.1%} > {self.thresholds['collision_rate_gap']:.1%}"
            )
            trigger_retrain = True
        
        # 6. Language Compliance Gap
        lc_gap = sim_metrics.language_compliance - real_metrics.language_compliance
        gaps["language_compliance_gap"] = lc_gap
        if lc_gap > self.thresholds["language_compliance_gap"]:
            trigger_reasons.append(
                f"Language compliance gap: {lc_gap:.1%} > {self.thresholds['language_compliance_gap']:.1%}"
            )
            trigger_retrain = True
        
        # Composite score (가중 평균)
        composite_score = sum(
            self.weights[k] * min(gaps.get(k, 0) / self.thresholds.get(k, 1), 1.0)
            for k in self.weights
        )
        
        return GapReport(
            timestamp=datetime.now().isoformat(),
            sim_metrics=sim_metrics,
            real_metrics=real_metrics,
            gaps=gaps,
            trigger_retrain=trigger_retrain,
            trigger_reasons=trigger_reasons,
            composite_score=composite_score,
        )
    
    def compute_metrics_from_db(
        self,
        db_path: str,
        episode_ids: List[str],
    ) -> ArmMetrics:
        """
        Episode DB 데이터 → ArmMetrics 계산
        """
        import sqlite3
        conn = sqlite3.connect(db_path)
        
        # Fetch episodes
        placeholders = ",".join("?" for _ in episode_ids)
        episodes = conn.execute(
            f"SELECT * FROM episodes WHERE episode_id IN ({placeholders})",
            episode_ids,
        ).fetchall()
        
        if not episodes:
            return ArmMetrics(0, 0, 0, 0, 0, 0)
        
        # Compute metrics
        successes = [e[5] for e in episodes]  # success column
        task_sr = sum(successes) / len(successes)
        
        # Frame-level metrics
        frames = conn.execute(
            f"SELECT * FROM frames WHERE episode_id IN ({placeholders})",
            episode_ids,
        ).fetchall()
        
        # Compute smoothness (jerk = derivative of acceleration)
        # Compute collision rate
        # Compute cycle times
        
        # Simplified placeholder
        return ArmMetrics(
            task_success_rate=task_sr,
            grasp_success_rate=0.9,
            trajectory_smoothness=100.0,
            cycle_time_ms=5000,
            collision_rate=0.02,
            language_compliance=0.95,
        )
```

---

## 4. Phase 9: Auto Retrain Pipeline (Arm VLA)

### 4.1 6단계 Retrain Pipeline

```
Trigger 발생 (Gap > Threshold)
       │
       ▼
┌──────────────────────┐
│ Step 1: Data Collect  │ ← 실패 에피소드 추출 + Sim에서 유사 시나리오 생성
│ "틀린 문제 수집"      │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Step 2: Fine-tune     │ ← VLA transformer LoRA fine-tuning
│ "취약점 집중 학습"    │ (hard-negative mining, curriculum learning)
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Step 3: Evaluate     │ ← Isaac Sim에서 100회 평가
│ "성능 확인"          │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Step 4: Export       │ ← .pt → .onnx → TensorRT .plan
│ "배포용으로 변환"     │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Step 5: Blue-Green   │ ← 무중단 배포
│ "안전하게 교체"       │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Step 6: Validate     │ ← 실제 로봇 10회 테스트
│ "최종 확인"          │
└──────────────────────┘
           │
      Success > 85%? ──Yes──→ 완료
           No
           │
           ▼
      Retrospect + 로깅
```

### 4.2 Auto Retrain 구현

```python
# src/digital_twin/auto_retrain_pipeline.py
"""
Arm VLA 자동 재학습 파이프라인
"""
import os
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Optional


class ArmVLARetrainPipeline:
    """
    Arm VLA 자동 재학습 6단계 파이프라인
    
    각 단계는 독립적으로 실행 가능하며,
    실패 시 이전 단계로 롤백
    """
    
    def __init__(
        self,
        project_root: str = ".",
        model_dir: str = "models/vla",
        data_dir: str = "data",
        config_path: str = "config/digital_twin_config.yaml",
    ):
        self.project_root = Path(project_root)
        self.model_dir = self.project_root / model_dir
        self.data_dir = self.project_root / data_dir
        self.current_pipeline = None
        self.log = []
    
    def run_pipeline(
        self,
        retrain_reason: str = "",
        focus_tasks: list = None,
        num_demonstrations: int = 20,
    ) -> bool:
        """
        전체 재학습 파이프라인 실행
        
        Args:
            retrain_reason: 재학습 트리거 사유
            focus_tasks: 집중 학습할 태스크 목록
            num_demonstrations: 수집할 추가 시연 수
            
        Returns:
            True if pipeline successful
        """
        pipeline_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_pipeline = pipeline_id
        self.log = []
        
        self._log(f"=== Retrain Pipeline Started: {pipeline_id} ===")
        self._log(f"Reason: {retrain_reason}")
        
        # Step 1: Data Collection
        self._log("Step 1/6: Data Collection...")
        if not self._step_collect_data(focus_tasks, num_demonstrations):
            self._log("FAILED: Data Collection")
            return False
        
        # Step 2: Fine-tune
        self._log("Step 2/6: VLA Fine-tuning...")
        if not self._step_finetune(pipeline_id):
            self._log("FAILED: Fine-tuning")
            return False
        
        # Step 3: Evaluate
        self._log("Step 3/6: Evaluation...")
        eval_result = self._step_evaluate(pipeline_id)
        if not eval_result:
            self._log("FAILED: Evaluation")
            return False
        
        # Step 4: Export
        self._log("Step 4/6: ONNX + TensorRT Export...")
        if not self._step_export(pipeline_id):
            self._log("FAILED: Export")
            return False
        
        # Step 5: Blue-Green Deploy
        self._log("Step 5/6: Blue-Green Deploy...")
        if not self._step_deploy(pipeline_id):
            self._log("FAILED: Deploy")
            return False
        
        # Step 6: Validate
        self._log("Step 6/6: Validation...")
        if not self._step_validate(pipeline_id):
            self._log("FAILED: Validation (rolling back)")
            self._rollback()
            return False
        
        self._log(f"=== Retrain Pipeline Complete: {pipeline_id} ===")
        return True
    
    def _step_collect_data(self, focus_tasks, num_demos) -> bool:
        """실패 에피소드 수집 + Isaac Sim 유사 시나리오 생성"""
        # 실패 에피소드 DB에서 추출
        # Isaac Sim에서 유사 조건 시나리오 100개 생성
        # CosmosTransfer로 이미지 변환
        time.sleep(1)  # Simulated
        return True
    
    def _step_finetune(self, pipeline_id: str) -> bool:
        """VLA LoRA fine-tuning"""
        cmd = [
            "python", "src/vla_policy/cosmos_policy_finetune.py",
            "--checkpoint", str(self.model_dir / "ppo/best_model.pt"),
            "--output", str(self.model_dir / pipeline_id),
            "--lora_rank", "16",
            "--epochs", "10",
            "--lr", "1e-4",
        ]
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True)
        return result.returncode == 0
    
    def _step_evaluate(self, pipeline_id: str) -> Optional[dict]:
        """Isaac Sim 평가"""
        cmd = [
            "python", "src/isaac_lab/evaluate_vla.py",
            "--checkpoint", str(self.model_dir / pipeline_id / "best_model.pt"),
            "--num_trials", "100",
        ]
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True)
        if result.returncode == 0:
            return json.loads(result.stdout)
        return None
    
    def _step_export(self, pipeline_id: str) -> bool:
        """ONNX + TensorRT 변환"""
        # ONNX export
        cmd_onnx = [
            "python", "src/vla_policy/export_onnx.py",
            "--checkpoint", str(self.model_dir / pipeline_id / "best_model.pt"),
            "--output", str(self.model_dir / pipeline_id / "vla_model.onnx"),
        ]
        if subprocess.run(cmd_onnx, cwd=self.project_root).returncode != 0:
            return False
        
        # TensorRT FP16
        cmd_trt = [
            "trtexec",
            f"--onnx={self.model_dir / pipeline_id / 'vla_model.onnx'}",
            f"--saveEngine={self.model_dir / pipeline_id / 'vla_model_fp16.plan'}",
            "--fp16", "--workspace=8192",
        ]
        return subprocess.run(cmd_trt, cwd=self.project_root).returncode == 0
    
    def _step_deploy(self, pipeline_id: str) -> bool:
        """Blue-Green 무중단 배포"""
        cmd = [
            "bash", "scripts/deploy_policy.sh",
            "--model", str(self.model_dir / pipeline_id),
            "--policy-id", pipeline_id,
        ]
        return subprocess.run(cmd, cwd=self.project_root).returncode == 0
    
    def _step_validate(self, pipeline_id: str) -> bool:
        """실제 로봇 10회 테스트"""
        cmd = [
            "python", "src/follower_arm/validate_policy.py",
            "--engine", str(self.model_dir / pipeline_id / "vla_model_fp16.plan"),
            "--num_trials", "10",
        ]
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True)
        if result.returncode == 0:
            validation = json.loads(result.stdout)
            return validation.get("success_rate", 0) > 0.85
        return False
    
    def _rollback(self):
        """이전 버전으로 롤백"""
        cmd = [
            "bash", "scripts/deploy_policy.sh",
            "--rollback",
        ]
        subprocess.run(cmd, cwd=self.project_root)
        self._log("ROLLED BACK to previous version")
    
    def _log(self, message: str):
        """파이프라인 로그 기록"""
        timestamped = f"[{datetime.now().strftime('%H:%M:%S')}] {message}"
        self.log.append(timestamped)
        print(timestamped)
```

### 4.3 Policy Registry

```python
# src/digital_twin/policy_registry.py
"""
VLA 정책 버전 관리 + Blue-Green 배포
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class VLApolicyRegistry:
    """
    VLA 정책 버전 관리
    
    - Active: 현재 사용 중인 정책
    - Backup: 이전 정책 (롤백용)
    - Staged: 배포 대기 중
    - Archived: 기록 보존
    """
    
    def __init__(self, registry_path: str = "models/policy_registry.json"):
        self.registry_path = Path(registry_path)
        self.registry = self._load()
    
    def _load(self) -> dict:
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {
            "active": None,
            "backup": None,
            "staged": [],
            "archived": [],
            "history": [],
        }
    
    def save(self):
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)
    
    def register(self, policy_id: str, model_path: str, metadata: dict):
        """새 정책 등록 (staged 상태)"""
        entry = {
            "policy_id": policy_id,
            "model_path": str(model_path),
            "created": datetime.now().isoformat(),
            "metadata": metadata,
            "status": "staged",
        }
        self.registry["staged"].append(entry)
        self.registry["history"].append(entry)
        self.save()
    
    def activate(self, policy_id: str):
        """Blue-Green 전환"""
        # Find staged policy
        staged = [p for p in self.registry["staged"] if p["policy_id"] == policy_id]
        if not staged:
            return False
        
        # Current active → backup
        if self.registry["active"]:
            self.registry["backup"] = self.registry["active"]
            self.registry["backup"]["status"] = "backup"
        
        # Staged → active
        staged[0]["status"] = "active"
        self.registry["active"] = staged[0]
        self.registry["staged"] = [p for p in self.registry["staged"] if p["policy_id"] != policy_id]
        
        # Symlink update for blue-green
        self._update_symlink(policy_id)
        
        self.save()
        return True
    
    def rollback(self):
        """Backup → Active 롤백"""
        if not self.registry["backup"]:
            return False
        
        backup = self.registry["backup"]
        if self.registry["active"]:
            self.registry["active"]["status"] = "archived"
            self.registry["archived"].append(self.registry["active"])
        
        backup["status"] = "active"
        self.registry["active"] = backup
        self.registry["backup"] = None
        
        self._update_symlink(backup["policy_id"])
        self.save()
        return True
    
    def _update_symlink(self, policy_id: str):
        """Blue-Green 심볼릭 링크 업데이트"""
        # models/vla/active → models/policy_id/ (atomic switch)
        active_link = Path("models/vla/active")
        target = Path(f"models/vla/{policy_id}")
        
        if active_link.exists() or active_link.is_symlink():
            active_link.unlink()
        
        os.symlink(target, active_link)
    
    def get_active(self) -> Optional[dict]:
        return self.registry.get("active")
    
    def get_backup(self) -> Optional[dict]:
        return self.registry.get("backup")
```

---

## 5. Phase 10: Orchestrator

### 5.1 State Machine

```
         ┌──────────┐
         │   IDLE    │ ← 초기 상태, 아무것도 하지 않음
         └─────┬─────┘
               │ start()
               ▼
         ┌──────────┐
         │MONITORING│ ← 5분 주기로 데이터 수집 + 메트릭 계산
         └─────┬─────┘
               │ analyze() called
               ▼
         ┌──────────┐
         │ ANALYZING│ ← Gap Analyzer 실행
         └─────┬─────┘
               │
      ┌────────┴────────┐
      │ Gap > Threshold │
     Yes               No
      │                 │
      ▼                 └──→ MONITORING
┌────────────┐
│ RETRAINING │ ← Auto Retrain Pipeline 실행
└──────┬─────┘
       │ complete
       ▼
┌──────────┐
│DEPLOYING │ ← Blue-Green 배포
└─────┬────┘
      │ done
      ▼
┌──────────┐
│MONITORING│ ← 재개
└──────────┘
```

### 5.2 Orchestrator 구현

```python
# src/digital_twin/orchestrator.py
"""
Digital Twin 중앙 오케스트레이터
"""
import time
import json
import threading
from enum import Enum
from datetime import datetime
from typing import Optional


class OrchestratorState(Enum):
    IDLE = "IDLE"
    MONITORING = "MONITORING"
    ANALYZING = "ANALYZING"
    RETRAINING = "RETRAINING"
    DEPLOYING = "DEPLOYING"
    ERROR = "ERROR"


class ArmDigitalTwinOrchestrator:
    """
    Arm VLA Digital Twin 오케스트레이터
    
    5분 주기로 모니터링 → 갭 분석 → 재학습 → 배포 순환
    """
    
    def __init__(
        self,
        config_path: str = "config/digital_twin_config.yaml",
        monitoring_interval: int = 300,  # 5분
        min_episodes_for_analysis: int = 10,
    ):
        self.config = self._load_config(config_path)
        self.monitoring_interval = monitoring_interval
        self.min_episodes = min_episodes_for_analysis
        
        self.state = OrchestratorState.IDLE
        self.cycle_count = 0
        self.last_analysis_time = 0
        self.last_retrain_time = 0
        
        # Components
        self.logger = None
        self.gap_analyzer = None
        self.retrain_pipeline = None
        self.policy_registry = None
        
        # Monitoring thread
        self.monitor_thread = None
        self.running = False
    
    def start(self):
        """Orchestrator 시작"""
        print("=== Digital Twin Orchestrator Started ===")
        self._init_components()
        self.state = OrchestratorState.MONITORING
        self.running = True
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        print(f"Monitoring interval: {self.monitoring_interval}s")
        print(f"Min episodes for analysis: {self.min_episodes}")
    
    def stop(self):
        """Orchestrator 중지"""
        self.running = False
        self.state = OrchestratorState.IDLE
        print("Orchestrator stopped")
    
    def _init_components(self):
        """컴포넌트 초기화"""
        from src.digital_twin.data_logger import ArmEpisodeLogger
        from src.digital_twin.gap_analyzer import ArmGapAnalyzer
        from src.digital_twin.auto_retrain_pipeline import ArmVLARetrainPipeline
        from src.digital_twin.policy_registry import VLApolicyRegistry
        
        self.logger = ArmEpisodeLogger(
            db_path=self.config.get("db_path", "data/episode_db.sqlite"),
        )
        self.gap_analyzer = ArmGapAnalyzer()
        self.retrain_pipeline = ArmVLARetrainPipeline()
        self.policy_registry = VLApolicyRegistry()
    
    def _monitor_loop(self):
        """메인 모니터링 루프"""
        while self.running:
            try:
                self._monitoring_cycle()
            except Exception as e:
                print(f"Error in monitoring cycle: {e}")
                self.state = OrchestratorState.ERROR
            
            time.sleep(self.monitoring_interval)
    
    def _monitoring_cycle(self):
        """단일 모니터링 사이클"""
        self.cycle_count += 1
        print(f"\n[Cycle {self.cycle_count}] State: {self.state.value}")
        
        if self.state != OrchestratorState.MONITORING:
            return
        
        # 에피소드 수 확인
        episode_count = self.logger.get_episode_count()
        success_rate = self.logger.get_success_rate()
        print(f"  Episodes: {episode_count}, SR: {success_rate:.1%}")
        
        if episode_count >= self.min_episodes:
            self.state = OrchestratorState.ANALYZING
            self._analyze_cycle()
    
    def _analyze_cycle(self):
        """갭 분석 사이클"""
        print("  → Analyzing sim-vs-real gap...")
        
        # Sim 메트릭 (Isaac Sim 결과)
        sim_metrics = self._get_sim_metrics()
        
        # Real 메트릭 (Episode DB)
        real_metrics = self.gap_analyzer.compute_metrics_from_db(
            db_path=self.config.get("db_path", "data/episode_db.sqlite"),
            episode_ids=self._get_recent_episodes(50),
        )
        
        # Gap 분석
        report = self.gap_analyzer.analyze(sim_metrics, real_metrics)
        
        print(f"  Composite Score: {report.composite_score:.3f}")
        for key, value in report.gaps.items():
            print(f"    {key}: {value:.3f}")
        
        if report.trigger_retrain:
            print(f"  🚨 Retrain triggered! Reasons:")
            for reason in report.trigger_reasons:
                print(f"    - {reason}")
            
            self.state = OrchestratorState.RETRAINING
            self._retrain_cycle(report)
        else:
            print("  ✅ No retrain needed")
            self.state = OrchestratorState.MONITORING
    
    def _retrain_cycle(self, gap_report):
        """재학습 사이클"""
        print("  → Running auto retrain pipeline...")
        
        success = self.retrain_pipeline.run_pipeline(
            retrain_reason="; ".join(gap_report.trigger_reasons),
            focus_tasks=["pick_and_place"],
            num_demonstrations=20,
        )
        
        if success:
            self.state = OrchestratorState.DEPLOYING
            self._deploy_cycle()
        else:
            print("  ✗ Retrain failed")
            self.state = OrchestratorState.ERROR
    
    def _deploy_cycle(self):
        """배포 사이클"""
        print("  → Deploying new policy...")
        
        latest_model = self.retrain_pipeline.current_pipeline
        if latest_model:
            self.policy_registry.register(
                policy_id=latest_model,
                model_path=f"models/vla/{latest_model}/vla_model_fp16.plan",
                metadata={
                    "retrain_reason": "automatic",
                    "cycle": self.cycle_count,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            self.policy_registry.activate(latest_model)
        
        self.state = OrchestratorState.MONITORING
        print("  ✅ Deploy complete, returning to monitoring")
    
    def _get_sim_metrics(self):
        """Isaac Sim 메트릭 조회"""
        baseline_path = "results/sim2real_eval.json"
        if os.path.exists(baseline_path):
            with open(baseline_path) as f:
                return json.load(f)
        return None
    
    def _get_recent_episodes(self, n: int):
        """최근 N개 에피소드 ID"""
        import sqlite3
        conn = sqlite3.connect(self.config.get("db_path", "data/episode_db.sqlite"))
        rows = conn.execute(
            "SELECT episode_id FROM episodes ORDER BY rowid DESC LIMIT ?", (n,)
        ).fetchall()
        return [r[0] for r in rows]
```

---

## 6. Configuration Reference

```yaml
# config/digital_twin_config.yaml
digital_twin:
  # Monitoring
  monitoring_interval: 300  # 5초 (초 단위)
  min_episodes_for_analysis: 10
  min_cycle_interval: 3600  # 최소 재학습 간격 (1시간)
  
  # Database
  db_path: "data/episode_db.sqlite"
  video_dir: "data/episodes"
  max_episodes: 10000  # 자동 정리 기준
  
  # Gap thresholds
  thresholds:
    task_success_rate_gap: 0.15    # 15%
    grasp_success_rate_gap: 0.10   # 10%
    trajectory_smoothness_gap: 2.0  # 2x
    cycle_time_gap: 1.3            # 1.3x
    collision_rate_gap: 0.05       # 5%
    language_compliance_gap: 0.10  # 10%
  
  # Retrain pipeline
  retrain:
    lora_rank: 16
    finetune_epochs: 10
    learning_rate: 1.0e-4
    num_sim_eval_trials: 100
    num_real_validation_trials: 10
    min_validation_success_rate: 0.85
  
  # Policy registry
  policy_registry_path: "models/policy_registry.json"
  model_dir: "models/vla"
```
