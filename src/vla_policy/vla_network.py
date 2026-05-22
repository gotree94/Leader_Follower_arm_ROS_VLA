"""
VLA Policy Network — Vision-Language-Action Transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class VLAConfig:
    vision_encoder: str = "resnet50"
    vision_pretrained: bool = True
    vision_feat_dim: int = 2048
    image_size: Tuple[int, int] = (224, 224)
    lang_embed_dim: int = 512
    leader_joint_dim: int = 7
    follower_joint_dim: int = 6
    joint_hidden_dim: int = 256
    gripper_dim: int = 2
    gripper_hidden_dim: int = 64
    action_dim: int = 7
    last_action_dim: int = 7
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 2048
    dropout: float = 0.1
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 100


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
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
        tgt2 = self.norm1(tgt)
        tgt2 = self.self_attn(tgt2, tgt2, tgt2)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn(tgt2, memory, memory)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt


class VLAEncoder(nn.Module):
    def __init__(self, config: VLAConfig):
        super().__init__()
        self.config = config
        import torchvision.models as models
        resnet = models.resnet50(weights="IMAGENET1K_V2" if config.vision_pretrained else None)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.vision_proj = nn.Conv2d(config.vision_feat_dim, config.d_model, 1)
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(64, config.d_model, 3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )
        self.leader_joint_encoder = nn.Sequential(
            nn.Linear(config.leader_joint_dim, config.joint_hidden_dim),
            nn.LayerNorm(config.joint_hidden_dim), nn.ReLU(),
            nn.Linear(config.joint_hidden_dim, config.d_model),
        )
        self.follower_joint_encoder = nn.Sequential(
            nn.Linear(config.follower_joint_dim, config.joint_hidden_dim),
            nn.LayerNorm(config.joint_hidden_dim), nn.ReLU(),
            nn.Linear(config.joint_hidden_dim, config.d_model),
        )
        self.gripper_encoder = nn.Sequential(
            nn.Linear(config.gripper_dim, config.gripper_hidden_dim), nn.ReLU(),
            nn.Linear(config.gripper_hidden_dim, config.d_model),
        )
        self.lang_proj = nn.Linear(config.lang_embed_dim, config.d_model)
        self.last_action_encoder = nn.Linear(config.last_action_dim, config.d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 8, config.d_model) * 0.02)

    def forward(self, leader_rgb, follower_rgb, depth, leader_joints, follower_joints,
                gripper_states, lang_embed, last_action):
        leader_feat = self.vision_proj(self.vision_encoder(leader_rgb)).mean(dim=[2, 3])
        follower_feat = self.vision_proj(self.vision_encoder(follower_rgb)).mean(dim=[2, 3])
        depth_feat = self.depth_encoder(depth).squeeze(-1).squeeze(-1)
        leader_joint_feat = self.leader_joint_encoder(leader_joints)
        follower_joint_feat = self.follower_joint_encoder(follower_joints)
        gripper_feat = self.gripper_encoder(gripper_states)
        lang_feat = self.lang_proj(lang_embed)
        last_action_feat = self.last_action_encoder(last_action)
        tokens = torch.stack([leader_feat, follower_feat, depth_feat, leader_joint_feat,
                              follower_joint_feat, gripper_feat, lang_feat, last_action_feat], dim=1)
        return tokens + self.pos_embed


class VLAPolicy(nn.Module):
    def __init__(self, config: VLAConfig):
        super().__init__()
        self.config = config
        self.encoder = VLAEncoder(config)
        self.transformer_blocks = nn.ModuleList([
            CrossAttentionBlock(config.d_model, config.nhead, config.dim_feedforward, config.dropout)
            for _ in range(config.num_layers)
        ])
        self.action_head = nn.Sequential(
            nn.Linear(config.d_model, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, config.action_dim),
        )
        self.value_head = nn.Sequential(
            nn.Linear(config.d_model, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 1),
        )
        self.success_head = nn.Sequential(
            nn.Linear(config.d_model, 64), nn.ReLU(),
            nn.Linear(64, 1), nn.Sigmoid(),
        )

    def forward(self, leader_rgb, follower_rgb, depth, leader_joints, follower_joints,
                gripper_states, lang_embed, last_action):
        tokens = self.encoder(leader_rgb, follower_rgb, depth, leader_joints, follower_joints,
                              gripper_states, lang_embed, last_action)
        for block in self.transformer_blocks:
            tokens = block(tokens, tokens)
        latent = tokens.mean(dim=1)
        action = torch.tanh(self.action_head(latent))
        value = self.value_head(latent)
        success = self.success_head(latent)
        return action, value, success

    def get_action(self, obs: dict, deterministic: bool = True) -> np.ndarray:
        with torch.no_grad():
            action, _, _ = self.forward(
                obs["leader_rgb"], obs["follower_rgb"], obs["depth"],
                obs["leader_joints"], obs["follower_joints"],
                obs["gripper_states"], obs["lang_embed"],
                obs.get("last_action", torch.zeros(1, 7)),
            )
            return action.cpu().numpy()
