"""
Digital Twin Orchestrator
"""
import time
import json
import os
import threading
from enum import Enum
from datetime import datetime


class OrchestratorState(Enum):
    IDLE = "IDLE"
    MONITORING = "MONITORING"
    ANALYZING = "ANALYZING"
    RETRAINING = "RETRAINING"
    DEPLOYING = "DEPLOYING"
    ERROR = "ERROR"


class ArmDigitalTwinOrchestrator:
    def __init__(self, config_path="config/digital_twin_config.yaml", monitoring_interval=300, min_episodes=10):
        self.config = self._load_config(config_path)
        self.monitoring_interval = monitoring_interval
        self.min_episodes = min_episodes
        self.state = OrchestratorState.IDLE
        self.cycle_count = 0
        self.logger = None
        self.gap_analyzer = None
        self.retrain_pipeline = None
        self.policy_registry = None
        self.monitor_thread = None
        self.running = False

    def _load_config(self, path):
        if os.path.exists(path):
            import yaml
            with open(path) as f:
                return yaml.safe_load(f)
        return {"db_path": "data/episode_db.sqlite"}

    def start(self):
        from src.digital_twin.data_logger import ArmEpisodeLogger
        from src.digital_twin.gap_analyzer import ArmGapAnalyzer
        from src.digital_twin.auto_retrain_pipeline import ArmVLARetrainPipeline
        from src.digital_twin.policy_registry import VLApolicyRegistry
        self.logger = ArmEpisodeLogger(db_path=self.config.get("db_path", "data/episode_db.sqlite"))
        self.gap_analyzer = ArmGapAnalyzer()
        self.retrain_pipeline = ArmVLARetrainPipeline()
        self.policy_registry = VLApolicyRegistry()
        self.state = OrchestratorState.MONITORING
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("Digital Twin Orchestrator started")

    def stop(self):
        self.running = False
        self.state = OrchestratorState.IDLE

    def _monitor_loop(self):
        while self.running:
            try:
                self._monitoring_cycle()
            except Exception as e:
                print(f"Error: {e}")
                self.state = OrchestratorState.ERROR
            time.sleep(self.monitoring_interval)

    def _monitoring_cycle(self):
        self.cycle_count += 1
        print(f"\n[Cycle {self.cycle_count}] State: {self.state.value}")
        if self.state != OrchestratorState.MONITORING:
            return
        count = self.logger.get_episode_count()
        sr = self.logger.get_success_rate()
        print(f"  Episodes: {count}, SR: {sr:.1%}")
        if count >= self.min_episodes:
            self.state = OrchestratorState.ANALYZING
            self._analyze_cycle()

    def _analyze_cycle(self):
        print("  → Analyzing...")
        sim = self.gap_analyzer.compute_metrics_from_db(self.config.get("db_path", "data/episode_db.sqlite"), ["dummy"])
        real = sim  # placeholder: 실제로는 separate metrics
        report = self.gap_analyzer.analyze(sim, real)
        print(f"  Composite: {report.composite_score:.3f}")
        if report.trigger_retrain:
            self.state = OrchestratorState.RETRAINING
            self._retrain_cycle(report)
        else:
            self.state = OrchestratorState.MONITORING

    def _retrain_cycle(self, report):
        print("  → Retraining...")
        success = self.retrain_pipeline.run_pipeline()
        if success:
            self.state = OrchestratorState.DEPLOYING
            self._deploy_cycle()
        else:
            self.state = OrchestratorState.ERROR

    def _deploy_cycle(self):
        print("  → Deploying...")
        pid = self.retrain_pipeline.current_pipeline
        if pid:
            self.policy_registry.register(pid, f"models/vla/{pid}/vla_model_fp16.plan", {"cycle": self.cycle_count})
            self.policy_registry.activate(pid)
        self.state = OrchestratorState.MONITORING


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    orch = ArmDigitalTwinOrchestrator()
    if args.start:
        orch.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            orch.stop()
