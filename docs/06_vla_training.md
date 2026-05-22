# VLA Policy Training with Isaac Lab

> Isaac Lab을 이용한 Vision-Language-Action Transformer 학습 — BC → PPO → Cosmos-Policy

---

## 1. VLA 아키텍처

### 1.1 전체 구조

```
Observation:
├── RGB Image (Leader view):    (3, 224, 224)      ← ResNet-50/ViT
├── RGB Image (Follower view):  (3, 224, 224)      ← ResNet-50/ViT
├── Depth Image:                (1, 224, 224)      ← Small CNN
├── Joint Positions (Leader):   (7,)               ← MLP (256→128)
├── Joint Positions (Follower): (6,)               ← MLP (256→128)
├── Gripper States:             (2,)               ← MLP (64→32)
├── Language Embedding:         (512,)             ← Cosmos Reason
└── Last Action:                (7,)               ← MLP (64→32)

Fusion:
└── Cross-Attention Transformer (4 layers, 8 heads, d_model=512)

Policy Heads:
├── Action Head: MLP(512→256→6)  ← Follower joint velocities
├── Gripper Head: MLP(512→64→1)  ← Gripper open/close
├── Value Head: MLP(512→256→1)   ← State value (for PPO)
└── Success Head: MLP(512→64→1)  ← Task success probability
```

### 1.2 Transformer Config

```python
# src/vla_policy/vla_network.py
"""
VLA (Vision-Language-Action) Transformer Network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class VLAConfig:
    """VLA Transformer 설정"""
    # Vision encoder
    vision_encoder: str = "resnet50"  # resnet50 or vit_base
    vision_pretrained: bool = True
    vision_feat_dim: int = 2048  # ResNet-50
    image_size: Tuple[int, int] = (224, 224)
    
    # Language encoder
    lang_embed_dim: int = 512
    
    # Joint encoder
    leader_joint_dim: int = 7
    follower_joint_dim: int = 6
    joint_hidden_dim: int = 256
    
    # Gripper encoder
    gripper_dim: int = 2
    gripper_hidden_dim: int = 64
    
    # Action
    action_dim: int = 7  # 6 joint vel + 1 gripper
    last_action_dim: int = 7
    
    # Transformer
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 2048
    dropout: float = 0.1
    
    # Training
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 100
    warmup_steps: int = 1000
    
    # PPO
    ppo_lr: float = 3e-5
    ppo_clip: float = 0.2
    ppo_epochs: int = 10
    ppo_batch_size: int = 4096
    gae_lambda: float = 0.95
    gamma: float = 0.99
    entropy_coef: float = 0.01
    value_coef: float = 0.5


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention fusion block
    
    여러 modality (vision, language, proprioception)를
    cross-attention으로 융합
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, tgt, memory):
        # Self-attention
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt2, tgt2, tgt2)[0]
        tgt = tgt + self.dropout(tgt2)
        
        # Cross-attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn(tgt2, memory, memory)[0]
        tgt = tgt + self.dropout(tgt2)
        
        # FFN
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        
        return tgt


class VLAEncoder(nn.Module):
    """
    VLA 멀티모달 인코더
    
    - Vision: ResNet-50 또는 ViT
    - Language: Cosmos Reason embedding
    - Proprioception: Joint + Gripper MLP
    """
    def __init__(self, config: VLAConfig):
        super().__init__()
        self.config = config
        
        # Vision encoder
        if "resnet" in config.vision_encoder:
            import torchvision.models as models
            resnet = models.resnet50(weights="IMAGENET1K_V2" if config.vision_pretrained else None)
            self.vision_encoder = nn.Sequential(*list(resnet.children())[:-2])
            self.vision_proj = nn.Conv2d(config.vision_feat_dim, config.d_model, 1)
        else:
            # ViT
            self.vision_encoder = nn.Identity()  # placeholder
            self.vision_proj = nn.Linear(768, config.d_model)
        
        # Depth encoder (small CNN)
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, config.d_model, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )
        
        # Joint encoders
        self.leader_joint_encoder = nn.Sequential(
            nn.Linear(config.leader_joint_dim, config.joint_hidden_dim),
            nn.LayerNorm(config.joint_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.joint_hidden_dim, config.d_model),
        )
        self.follower_joint_encoder = nn.Sequential(
            nn.Linear(config.follower_joint_dim, config.joint_hidden_dim),
            nn.LayerNorm(config.joint_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.joint_hidden_dim, config.d_model),
        )
        
        # Gripper encoder
        self.gripper_encoder = nn.Sequential(
            nn.Linear(config.gripper_dim, config.gripper_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.gripper_hidden_dim, config.d_model),
        )
        
        # Language encoder (projection only — embedding from Cosmos Reason)
        self.lang_proj = nn.Linear(config.lang_embed_dim, config.d_model)
        
        # Last action encoder
        self.last_action_encoder = nn.Linear(config.last_action_dim, config.d_model)
        
        # Position encoding (learned)
        self.pos_embed = nn.Parameter(torch.randn(1, 8, config.d_model) * 0.02)
        
    def forward(
        self,
        leader_rgb: torch.Tensor,      # (B, 3, 224, 224)
        follower_rgb: torch.Tensor,    # (B, 3, 224, 224)
        depth: torch.Tensor,           # (B, 1, 224, 224)
        leader_joints: torch.Tensor,   # (B, 7)
        follower_joints: torch.Tensor, # (B, 6)
        gripper_states: torch.Tensor,  # (B, 2)
        lang_embed: torch.Tensor,      # (B, 512)
        last_action: torch.Tensor,     # (B, 7)
    ) -> torch.Tensor:
        """멀티모달 인코딩 → fused latent (B, 8, d_model)"""
        
        # Vision
        leader_feat = self.vision_encoder(leader_rgb)  # (B, 2048, 7, 7)
        leader_feat = self.vision_proj(leader_feat)    # (B, d_model, 7, 7)
        leader_feat = leader_feat.mean(dim=[2, 3])     # (B, d_model) - global avg pool
        
        follower_feat = self.vision_encoder(follower_rgb)
        follower_feat = self.vision_proj(follower_feat)
        follower_feat = follower_feat.mean(dim=[2, 3])
        
        # Depth
        depth_feat = self.depth_encoder(depth).squeeze(-1).squeeze(-1)  # (B, d_model)
        
        # Proprioception
        leader_joint_feat = self.leader_joint_encoder(leader_joints)    # (B, d_model)
        follower_joint_feat = self.follower_joint_encoder(follower_joints)
        gripper_feat = self.gripper_encoder(gripper_states)
        last_action_feat = self.last_action_encoder(last_action)
        
        # Language
        lang_feat = self.lang_proj(lang_embed)
        
        # Stack all tokens (8 tokens)
        tokens = torch.stack([
            leader_feat,          # 0
            follower_feat,        # 1
            depth_feat,           # 2
            leader_joint_feat,    # 3
            follower_joint_feat,  # 4
            gripper_feat,         # 5
            lang_feat,            # 6
            last_action_feat,     # 7
        ], dim=1)  # (B, 8, d_model)
        
        # Add position encoding
        tokens = tokens + self.pos_embed
        
        return tokens


class VLAPolicy(nn.Module):
    """
    VLA Transformer 정책
    
    입력: 멀티모달 관측 (vision + language + proprioception)
    출력: Joint velocity + Gripper command
    """
    def __init__(self, config: VLAConfig):
        super().__init__()
        self.config = config
        self.encoder = VLAEncoder(config)
        
        # Transformer blocks (cross-attention fusion)
        self.transformer_blocks = nn.ModuleList([
            CrossAttentionBlock(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])
        
        # Policy heads
        self.action_head = nn.Sequential(
            nn.Linear(config.d_model, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, config.action_dim),
        )
        
        # Value head (for PPO)
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        
        # Success prediction head
        self.success_head = nn.Sequential(
            nn.Linear(config.d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        # Action noise (for exploration during training)
        self.action_noise = None
        
    def forward(self, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            action: (B, 7) — [6 joint vel, 1 gripper]
            value: (B, 1)
            success_prob: (B, 1)
        """
        # Encode
        tokens = self.encoder(*args, **kwargs)  # (B, 8, d_model)
        
        # Transformer fusion
        for block in self.transformer_blocks:
            tokens = block(tokens, tokens)
        
        # Average tokens for global representation
        latent = tokens.mean(dim=1)  # (B, d_model)
        
        # Heads
        action = self.action_head(latent)  # (B, 7)
        value = self.value_head(latent)    # (B, 1)
        success = self.success_head(latent)  # (B, 1)
        
        # Tanh for bounded action
        action = torch.tanh(action)  # [-1, 1] → scale to joint limits
        
        return action, value, success
    
    def get_action(self, obs: dict, deterministic: bool = True) -> np.ndarray:
        """환경 추론용 간편 함수"""
        with torch.no_grad():
            action, _, _ = self.forward(
                leader_rgb=obs["leader_rgb"],
                follower_rgb=obs["follower_rgb"],
                depth=obs["depth"],
                leader_joints=obs["leader_joints"],
                follower_joints=obs["follower_joints"],
                gripper_states=obs["gripper_states"],
                lang_embed=obs["lang_embed"],
                last_action=obs.get("last_action", torch.zeros(1, 7)),
            )
            
            if not deterministic and self.action_noise:
                action = action + self.action_noise.sample()
            
            return action.cpu().numpy()
    
    def load_cosmos_policy(self, checkpoint_path: str):
        """
        Cosmos-Policy 사전 학습 가중치 로드
        
        Transformer backbone만 로드하고,
        task-specific heads (action/value/success)는
        랜덤 초기화 유지
        """
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # Load transformer weights only
        model_state = self.state_dict()
        for name, param in checkpoint.items():
            if "transformer" in name and name in model_state:
                model_state[name].copy_(param)
        
        self.load_state_dict(model_state)
        print(f"Loaded Cosmos-Policy backbone from {checkpoint_path}")
```

---

## 2. Isaac Lab VLA 태스크 환경

### 2.1 환경 설정

```python
# src/isaac_lab/arm_vla_cfg.py
"""
Isaac Lab VLA 태스크 설정
"""
from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass
class ArmVLAConfig:
    """VLA 학습 환경 설정"""
    
    # Scene
    leader_usd: str = "urdf/franka_panda.usd"
    follower_usd: str = "urdf/ur5e.usd"
    table_height: float = 0.7
    
    # Observation space
    image_width: int = 224
    image_height: int = 224
    
    # Action space
    action_scale: float = 1.0
    joint_velocity_scale: float = 0.5  # rad/s
    gripper_scale: float = 1.0
    
    # Joint limits (for action denormalization)
    leader_joint_limits: List[Tuple[float, float]] = field(default_factory=lambda: [
        (-2.897, 2.897), (-1.762, 1.762), (-2.897, 2.897),
        (-3.071, 3.071), (-2.897, 2.897), (-0.017, 3.752),
        (-2.897, 2.897),
    ])
    follower_joint_limits: List[Tuple[float, float]] = field(default_factory=lambda: [
        (-6.283, 6.283), (-6.283, 6.283), (-6.283, 6.283),
        (-6.283, 6.283), (-6.283, 6.283), (-6.283, 6.283),
    ])
    
    # Reward
    task_reward: float = 10.0
    collision_penalty: float = -1.0
    smoothness_weight: float = -0.01
    time_penalty: float = -0.01
    
    # Episode
    episode_length: int = 150  # timesteps (5초 @ 30Hz)
    num_envs: int = 64  # 병렬 환경 수
    seed: int = 42
    
    # Domain randomization
    randomize_lighting: bool = True
    randomize_physics: bool = True
    randomize_camera: bool = True
```

### 2.2 태스크 정의

```python
# src/isaac_lab/arm_vla_task.py
"""
Isaac Lab VLA Manipulation Task
"""
import torch
import numpy as np
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg, EventTermCfg
from omni.isaac.lab.assets import Artication, RigidObject
from omni.isaac.lab.utils.math import quat_from_euler


class ArmVLATask(ManagerBasedRLEnv):
    """
    VLA Leader-Follower Arm Task
    
    Leader Arm: Franka Panda (시연 / demonstration replay)
    Follower Arm: UR5e (VLA 정책 학습)
    """
    
    def __init__(self, cfg: ArmVLAConfig):
        super().__init__(cfg)
        self.cfg = cfg
        
        # Task state
        self.task_type = "pick_and_place"
        self.task_progress = 0.0
        self.target_obj_pos = None
        self.target_place_pos = None
    
    def _get_observations(self) -> dict:
        """관측 공간 구성"""
        obs = {}
        
        # RGB-D images (render cameras)
        obs["leader_rgb"] = self.get_camera_data("leader_rgb")
        obs["follower_rgb"] = self.get_camera_data("follower_rgb")
        obs["depth"] = self.get_camera_data("depth")
        
        # Joint states
        leader_joints = self.leader_arm.get_joint_positions()
        follower_joints = self.follower_arm.get_joint_positions()
        obs["leader_joints"] = leader_joints
        obs["follower_joints"] = follower_joints
        
        # Gripper states
        obs["gripper_states"] = torch.cat([
            self.leader_arm.get_joint_positions()[-2:].mean(),
            self.follower_arm.get_joint_positions()[-2:].mean(),
        ]).unsqueeze(0)
        
        # Language embedding (from Cosmos Reason or fixed for task)
        obs["lang_embed"] = self._get_language_embedding()
        
        # Last action (for smoothing)
        obs["last_action"] = self.last_action
        
        return obs
    
    def _compute_reward(self) -> torch.Tensor:
        """보상 함수 계산"""
        rewards = torch.zeros(self.num_envs, device=self.device)
        
        # 1. Task progress reward
        progress = self._compute_task_progress()
        rewards += progress * self.cfg.task_reward
        
        # 2. Success bonus
        success = self._check_success()
        rewards[success] += self.cfg.task_reward * 2
        
        # 3. Collision penalty
        collision = self._check_collision()
        rewards[collision] += self.cfg.collision_penalty
        
        # 4. Smoothness penalty (prevent jittery motion)
        action_diff = self.actions - self.last_action
        smoothness_penalty = self.cfg.smoothness_weight * torch.norm(action_diff, dim=1)
        rewards += smoothness_penalty
        
        # 5. Time penalty
        rewards += self.cfg.time_penalty
        
        return rewards
    
    def _compute_task_progress(self) -> torch.Tensor:
        """태스크 진행도 계산 (0.0 ~ 1.0)"""
        if self.task_type == "pick_and_place":
            return self._compute_pick_and_place_progress()
        return torch.zeros(self.num_envs, device=self.device)
    
    def _compute_pick_and_place_progress(self) -> torch.Tensor:
        """
        Pick-and-Place 진행도:
        0.0: Initial
        0.3: Gripper near object
        0.6: Object grasped
        0.8: Object above target
        1.0: Object placed
        """
        progress = torch.zeros(self.num_envs, device=self.device)
        
        # Distance from gripper to object
        ee_pos = self.follower_arm.get_ee_pose()[:, :3]
        obj_pos = self.target_obj_pos
        dist_to_obj = torch.norm(ee_pos - obj_pos, dim=1)
        
        # Distance from object to target
        dist_to_target = torch.norm(obj_pos - self.target_place_pos, dim=1)
        
        # Gripper state (0=open, 1=closed)
        gripper_state = self.follower_arm.get_joint_positions()[:, -2:].mean(dim=1)
        
        # Progress logic
        near_obj = dist_to_obj < 0.05
        grasped = near_obj & (gripper_state > 0.8)
        above_target = dist_to_target < 0.05
        placed = above_target & (gripper_state < 0.2)
        
        progress[near_obj] = 0.3
        progress[grasped] = 0.6
        progress[above_target & grasped] = 0.8
        progress[placed] = 1.0
        
        return progress
    
    def _check_success(self) -> torch.Tensor:
        """태스크 성공 여부"""
        progress = self._compute_task_progress()
        return progress >= 1.0
    
    def _check_collision(self) -> torch.Tensor:
        """충돌 감지"""
        # Check self-collision and environment collision
        collision_data = self.get_collision_data()
        return collision_data["has_collision"]
```

---

## 3. 훈련 실행

### 3.1 BC 사전 학습

```python
# src/isaac_lab/train_vla.py
"""
VLA Trainer — Behavior Cloning + PPO
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from vla_policy.vla_network import VLAPolicy, VLAConfig
import json


class TrajectoryDataset(Dataset):
    """
    Leader 시연 데이터 → PyTorch Dataset
    
    CosmosWriter가 생성한 SDG 데이터를 로드
    """
    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.split = split
        self.episodes = self._load_episodes()
        self.frames = self._build_frame_index()
        
    def _load_episodes(self):
        """에피소드 목록 로드"""
        episodes = []
        split_file = self.data_dir / f"{self.split}_episodes.json"
        
        if split_file.exists():
            with open(split_file) as f:
                episodes = json.load(f)
        else:
            # 모든 에피소드 스캔
            ep_dirs = sorted(self.data_dir.glob("ep_*"))
            episodes = [str(d.name) for d in ep_dirs]
        
        return episodes
    
    def _build_frame_index(self):
        """전체 프레임 인덱스 구축"""
        frames = []
        for ep in self.episodes:
            meta_path = self.data_dir / ep / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                for i, frame in enumerate(meta["frames"]):
                    frames.append((ep, i))
        return frames
    
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        ep_name, frame_idx = self.frames[idx]
        meta_path = self.data_dir / ep_name / "metadata.json"
        
        with open(meta_path) as f:
            meta = json.load(f)
        
        frame = meta["frames"][frame_idx]
        
        # Load data
        sample = {
            "leader_joints": torch.tensor(frame["leader_joints"], dtype=torch.float32),
            "follower_joints": torch.tensor(frame["follower_joints"], dtype=torch.float32),
            "leader_gripper": torch.tensor([frame["leader_gripper"]], dtype=torch.float32),
            "follower_gripper": torch.tensor([frame["follower_gripper"]], dtype=torch.float32),
            "instruction": frame.get("instruction", ""),
            "success": torch.tensor([frame.get("success", False)], dtype=torch.float32),
        }
        return sample


class BCVLAEncoder(Dataset):
    """
    VLA 모델 입력 형태로 데이터 변환
    """
    def __init__(self, dataset: TrajectoryDataset, config: VLAConfig):
        self.dataset = dataset
        self.config = config
        self.lang_vocab = {}  # 언어 임베딩 캐시
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        
        # Dummy image data (실제로는 RGB/Depth 파일에서 로드)
        leader_rgb = torch.zeros(3, *self.config.image_size)
        follower_rgb = torch.zeros(3, *self.config.image_size)
        depth = torch.zeros(1, *self.config.image_size)
        
        # Language embedding (Cosmos Reason으로 실제 변환 필요)
        lang_embed = torch.zeros(self.config.lang_embed_dim)
        
        # Joint + gripper
        leader_joints = sample["leader_joints"][:7]
        follower_joints = sample["follower_joints"][:6]
        gripper_states = torch.stack([sample["leader_gripper"], sample["follower_gripper"]]).squeeze()
        last_action = torch.cat([follower_joints, sample["follower_gripper"]])
        
        # Action target (follower joints + gripper)
        action_target = torch.cat([follower_joints, sample["follower_gripper"]])
        
        return {
            "leader_rgb": leader_rgb,
            "follower_rgb": follower_rgb,
            "depth": depth,
            "leader_joints": leader_joints,
            "follower_joints": follower_joints,
            "gripper_states": gripper_states,
            "lang_embed": lang_embed,
            "last_action": last_action,
            "action_target": action_target,
        }


def train_bc(
    model: VLAPolicy,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: VLAConfig,
    device: str = "cuda",
    output_dir: str = "models/vla_bc",
):
    """
    Behavior Cloning 학습
    
    Loss: MSE (joint velocity) + BCE (gripper)
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.max_epochs
    )
    
    mse_loss = nn.MSELoss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    best_val_loss = float("inf")
    patience = 10
    patience_counter = 0
    
    for epoch in range(config.max_epochs):
        # Train
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            # Move to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward
            actions, _, success = model(
                batch["leader_rgb"],
                batch["follower_rgb"],
                batch["depth"],
                batch["leader_joints"],
                batch["follower_joints"],
                batch["gripper_states"],
                batch["lang_embed"],
                batch["last_action"],
            )
            
            # Loss
            loss_mse = mse_loss(actions[:, :6], batch["action_target"][:, :6])
            loss_gripper = bce_loss(actions[:, 6:], batch["action_target"][:, 6:])
            loss = loss_mse + loss_gripper
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        scheduler.step()
        
        # Validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                actions, _, _ = model(
                    batch["leader_rgb"],
                    batch["follower_rgb"],
                    batch["depth"],
                    batch["leader_joints"],
                    batch["follower_joints"],
                    batch["gripper_states"],
                    batch["lang_embed"],
                    batch["last_action"],
                )
                loss = mse_loss(actions[:, :6], batch["action_target"][:, :6])
                val_loss += loss.item()
        val_loss /= len(val_loader)
        
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), f"{output_dir}/best_model.pt")
            print(f"  → Saved best model (val_loss={val_loss:.6f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    return model


def train_ppo(env, model, config):
    """
    PPO 강화학습 Fine-tuning
    
    BC로 사전 학습된 정책을 PPO로 추가 최적화
    """
    from stable_baselines3 import PPO as SB3PPO
    from stable_baselines3.common.env_util import make_vec_env
    
    # Wrap Isaac Lab env for SB3
    # (실제로는 Isaac Lab의 RL trainer 사용)
    
    print("PPO training initialized")
    print(f"  Env num: {config.num_envs}")
    print(f"  Batch size: {config.ppo_batch_size}")
    print(f"  Learning rate: {config.ppo_lr}")
    
    return model
```

### 3.2 훈련 실행 스크립트

```bash
# scripts/run_vla_training.sh
#!/bin/bash
# VLA 학습 실행 스크립트

set -e

# Conda 환경
CONDA_BASE=$(conda info --base)
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate isaaclab

# 설정
CONFIG="config/vla_training_config.yaml"
OUTPUT_DIR="results/vla_training"
DATA_DIR="data/cosmos_sdg"
MODEL_DIR="models/vla"

mkdir -p $OUTPUT_DIR $MODEL_DIR

# Phase 1: BC Pretraining
echo "=== Phase 1: Behavior Cloning ==="
python src/isaac_lab/train_vla.py \
    --mode bc \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --batch_size 256 \
    --epochs 100 \
    --lr 3e-4 \
    --model_dir $MODEL_DIR/bc

# Phase 2: PPO Fine-tuning
echo "=== Phase 2: PPO Fine-tuning ==="
python src/isaac_lab/train_vla.py \
    --mode ppo \
    --checkpoint $MODEL_DIR/bc/best_model.pt \
    --output_dir $OUTPUT_DIR \
    --num_envs 64 \
    --ppo_epochs 10 \
    --ppo_batch_size 4096 \
    --model_dir $MODEL_DIR/ppo

# Phase 3: ONNX Export
echo "=== Phase 3: ONNX Export ==="
python src/vla_policy/export_onnx.py \
    --checkpoint $MODEL_DIR/ppo/best_model.pt \
    --output $MODEL_DIR/vla_model.onnx

# Phase 4: TensorRT Build
echo "=== Phase 4: TensorRT Engine ==="
/usr/src/tensorrt/bin/trtexec \
    --onnx=$MODEL_DIR/vla_model.onnx \
    --saveEngine=$MODEL_DIR/vla_model_fp16.plan \
    --fp16 \
    --workspace=8192 \
    --minShapes=leader_rgb:1x3x224x224 \
    --optShapes=leader_rgb:4x3x224x224 \
    --maxShapes=leader_rgb:8x3x224x224

echo "=== Training Complete ==="
echo "Models saved to: $MODEL_DIR"
ls -lh $MODEL_DIR/
```

---

## 4. Cosmos-Policy Post-Training

### 4.1 LoRA Fine-tuning

```python
# src/vla_policy/cosmos_policy_finetune.py
"""
Cosmos-Policy LoRA Fine-tuning
"""
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM
from vla_policy.vla_network import VLAConfig


def setup_cosmos_policy_lora(
    base_model_path: str = "nvidia/cosmos-predict2-2b",
    lora_rank: int = 16,
    lora_alpha: int = 32,
    target_modules: list = None,
):
    """
    Cosmos-Policy에 LoRA 적용
    
    Args:
        base_model_path: Cosmos-Policy base model
        lora_rank: LoRA rank (r)
        lora_alpha: LoRA scaling alpha
        target_modules: 적용할 module names
    """
    if target_modules is None:
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # → Trainable params: 0.15% of total
    
    return model


def finetune_cosmos_policy(
    model,
    train_dataset,
    val_dataset,
    config: VLAConfig,
    output_dir: str = "models/cosmos_policy_lora",
):
    """
    Cosmos-Policy LoRA Fine-tuning
    
    학습 가능한 파라미터는 LoRA adapter만 (≈0.15%)
    """
    from transformers import TrainingArguments, Trainer
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=8,
        num_train_epochs=10,
        learning_rate=1e-4,
        fp16=True,
        save_steps=500,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=500,
        save_total_limit=3,
        remove_unused_columns=False,
        report_to="tensorboard",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    trainer.train()
    
    # Save LoRA adapter only
    model.save_pretrained(f"{output_dir}/lora_adapter")
    print(f"LoRA adapter saved to {output_dir}/lora_adapter")
    
    return model
```
