"""
Gap Analyzer — Sim-vs-Real 성능 차이 분석
"""
import json
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class ArmMetrics:
    task_success_rate: float
    grasp_success_rate: float
    trajectory_smoothness: float
    cycle_time_ms: float
    collision_rate: float
    language_compliance: float


@dataclass
class GapReport:
    timestamp: str
    sim_metrics: ArmMetrics
    real_metrics: ArmMetrics
    gaps: Dict[str, float]
    trigger_retrain: bool
    trigger_reasons: List[str]
    composite_score: float


class ArmGapAnalyzer:
    def __init__(self):
        self.thresholds = {
            "task_success_rate_gap": 0.15,
            "grasp_success_rate_gap": 0.10,
            "trajectory_smoothness_gap": 2.0,
            "cycle_time_gap": 1.3,
            "collision_rate_gap": 0.05,
            "language_compliance_gap": 0.10,
        }
        self.weights = {
            "task_success_rate_gap": 0.40, "grasp_success_rate_gap": 0.15,
            "trajectory_smoothness_gap": 0.10, "cycle_time_gap": 0.05,
            "collision_rate_gap": 0.20, "language_compliance_gap": 0.10,
        }

    def analyze(self, sim: ArmMetrics, real: ArmMetrics) -> GapReport:
        gaps = {}
        trigger_reasons = []
        gap = sim.task_success_rate - real.task_success_rate
        gaps["task_success_rate_gap"] = gap
        if gap > self.thresholds["task_success_rate_gap"]:
            trigger_reasons.append(f"Task SR gap: {gap:.1%}")

        gap2 = sim.grasp_success_rate - real.grasp_success_rate
        gaps["grasp_success_rate_gap"] = gap2
        if gap2 > self.thresholds["grasp_success_rate_gap"]:
            trigger_reasons.append(f"Grasp SR gap: {gap2:.1%}")

        sm_gap = sim.trajectory_smoothness / max(real.trajectory_smoothness, 1e-6)
        gaps["trajectory_smoothness_gap"] = sm_gap
        if sm_gap > self.thresholds["trajectory_smoothness_gap"]:
            trigger_reasons.append(f"Smoothness gap: {sm_gap:.2f}x")

        ct_gap = real.cycle_time_ms / max(sim.cycle_time_ms, 1e-6)
        gaps["cycle_time_gap"] = ct_gap
        if ct_gap > self.thresholds["cycle_time_gap"]:
            trigger_reasons.append(f"Cycle time gap: {ct_gap:.2f}x")

        col_gap = real.collision_rate - sim.collision_rate
        gaps["collision_rate_gap"] = col_gap
        if col_gap > self.thresholds["collision_rate_gap"]:
            trigger_reasons.append(f"Collision gap: {col_gap:.1%}")

        lc_gap = sim.language_compliance - real.language_compliance
        gaps["language_compliance_gap"] = lc_gap
        if lc_gap > self.thresholds["language_compliance_gap"]:
            trigger_reasons.append(f"Language compliance gap: {lc_gap:.1%}")

        composite = sum(self.weights[k] * min(abs(gaps.get(k, 0)) / self.thresholds.get(k, 1), 1.0) for k in self.weights)

        return GapReport(
            timestamp=datetime.now().isoformat(), sim_metrics=sim, real_metrics=real,
            gaps=gaps, trigger_retrain=len(trigger_reasons) > 0,
            trigger_reasons=trigger_reasons, composite_score=composite)

    def compute_metrics_from_db(self, db_path: str, episode_ids: List[str]) -> ArmMetrics:
        import sqlite3
        conn = sqlite3.connect(db_path)
        eps = conn.execute(f"SELECT success FROM episodes WHERE episode_id IN ({','.join('?'*len(episode_ids))})", episode_ids).fetchall()
        sr = sum(e[0] for e in eps) / max(len(eps), 1)
        return ArmMetrics(task_success_rate=float(sr), grasp_success_rate=0.9,
                          trajectory_smoothness=100.0, cycle_time_ms=5000,
                          collision_rate=0.02, language_compliance=0.95)
