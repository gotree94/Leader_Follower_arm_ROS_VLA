"""
Digital Twin Data Logger — Arm Episode Logger
"""
import sqlite3
import json
import os
import time
import threading
import numpy as np
from datetime import datetime
from queue import Queue
from typing import Optional


class ArmEpisodeLogger:
    def __init__(self, db_path="data/episode_db.sqlite", video_dir="data/episodes", buffer_size=100):
        self.db_path = db_path
        self.video_dir = video_dir
        self.buffer = Queue()
        self.current_episode = None
        self.frame_count = 0
        self._init_db()
        os.makedirs(video_dir, exist_ok=True)
        print(f"ArmEpisodeLogger initialized: {db_path}")

    def _init_db(self):
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY, timestamp TEXT, arm_type TEXT,
                task_type TEXT, instruction TEXT, success BOOLEAN DEFAULT 0,
                duration_ms INTEGER, num_frames INTEGER, sim_origin BOOLEAN DEFAULT 0,
                model_version TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
            CREATE TABLE IF NOT EXISTS frames (
                frame_id INTEGER PRIMARY KEY AUTOINCREMENT, episode_id TEXT,
                timestep INTEGER, timestamp REAL, leader_joint_positions TEXT,
                follower_joint_positions TEXT, leader_gripper REAL, follower_gripper REAL,
                leader_ee_pose TEXT, follower_ee_pose TEXT, vla_action TEXT,
                vla_value REAL, vla_success_prob REAL, reward REAL DEFAULT 0.0,
                task_progress REAL DEFAULT 0.0, collision BOOLEAN DEFAULT 0,
                leader_rgb_path TEXT, follower_rgb_path TEXT,
                FOREIGN KEY (episode_id) REFERENCES episodes(episode_id));
            CREATE INDEX IF NOT EXISTS idx_frames_episode ON frames(episode_id);
        """)

    def start_episode(self, episode_id, task_type="unknown", instruction="", arm_type="follower",
                      model_version="unknown", sim_origin=False):
        self.current_episode = {
            "episode_id": episode_id, "timestamp": datetime.now().isoformat(),
            "arm_type": arm_type, "task_type": task_type, "instruction": instruction,
            "model_version": model_version, "sim_origin": sim_origin,
            "start_time": time.time(), "frames": []}
        self.frame_count = 0

    def log_frame(self, timestep, timestamp, leader_joints=None, follower_joints=None,
                  leader_gripper=0.0, follower_gripper=0.0, vla_action=None,
                  reward=0.0, task_progress=0.0, collision=False, **kwargs):
        if self.current_episode is None:
            return
        frame = {
            "timestep": timestep, "timestamp": timestamp,
            "leader_joint_positions": leader_joints[:7].tolist() if leader_joints is not None else None,
            "follower_joint_positions": follower_joints[:6].tolist() if follower_joints is not None else None,
            "leader_gripper": float(leader_gripper), "follower_gripper": float(follower_gripper),
            "vla_action": vla_action.tolist() if vla_action is not None else None,
            "reward": float(reward), "task_progress": float(task_progress), "collision": int(collision),
        }
        self.current_episode["frames"].append(frame)
        self.frame_count += 1

    def end_episode(self, success):
        if self.current_episode is None:
            return
        duration = (time.time() - self.current_episode["start_time"]) * 1000
        self.conn.execute(
            "INSERT OR REPLACE INTO episodes VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (self.current_episode["episode_id"], self.current_episode["timestamp"],
             self.current_episode["arm_type"], self.current_episode["task_type"],
             self.current_episode["instruction"], int(success), int(duration),
             self.frame_count, int(self.current_episode["sim_origin"]),
             self.current_episode["model_version"], datetime.now().isoformat()))
        frames = [(self.current_episode["episode_id"], f["timestep"], f["timestamp"],
                   json.dumps(f["leader_joint_positions"]), json.dumps(f["follower_joint_positions"]),
                   f["leader_gripper"], f["follower_gripper"], f.get("vla_action"),
                   f["reward"], f["task_progress"], f["collision"]) for f in self.current_episode["frames"]]
        self.conn.executemany(
            "INSERT INTO frames (episode_id, timestep, timestamp, leader_joint_positions, follower_joint_positions, leader_gripper, follower_gripper, vla_action, reward, task_progress, collision) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            frames)
        self.conn.commit()
        print(f"Episode saved: {self.current_episode['episode_id']} ({self.frame_count} frames)")
        self.current_episode = None
        self.frame_count = 0

    def get_episode_count(self):
        return self.conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]

    def get_success_rate(self, task_type=None):
        if task_type:
            return self.conn.execute("SELECT AVG(success) FROM episodes WHERE task_type=?", (task_type,)).fetchone()[0] or 0.0
        return self.conn.execute("SELECT AVG(success) FROM episodes").fetchone()[0] or 0.0
