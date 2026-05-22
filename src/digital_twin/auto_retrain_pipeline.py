"""
Auto Retrain Pipeline — VLA 자동 재학습
"""
import os
import subprocess
import json
import time
from pathlib import Path
from datetime import datetime


class ArmVLARetrainPipeline:
    def __init__(self, project_root=".", model_dir="models/vla", data_dir="data"):
        self.project_root = Path(project_root)
        self.model_dir = self.project_root / model_dir
        self.data_dir = self.project_root / data_dir
        self.current_pipeline = None
        self.log = []

    def run_pipeline(self, retrain_reason="", focus_tasks=None, num_demonstrations=20) -> bool:
        pipeline_id = f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.current_pipeline = pipeline_id
        self.log = []
        self._log(f"=== Retrain Pipeline: {pipeline_id} ===")

        self._log("Step 1/6: Data Collection...")
        if not self._step_collect_data(focus_tasks, num_demonstrations):
            return False

        self._log("Step 2/6: Fine-tuning...")
        if not self._step_finetune(pipeline_id):
            return False

        self._log("Step 3/6: Evaluation...")
        if not self._step_evaluate(pipeline_id):
            return False

        self._log("Step 4/6: Export...")
        if not self._step_export(pipeline_id):
            return False

        self._log("Step 5/6: Deploy...")
        if not self._step_deploy(pipeline_id):
            return False

        self._log("Step 6/6: Validation...")
        if not self._step_validate(pipeline_id):
            self._rollback()
            return False

        self._log(f"=== Complete: {pipeline_id} ===")
        return True

    def _step_collect_data(self, focus_tasks, num_demos):
        time.sleep(0.5)
        return True

    def _step_finetune(self, pipeline_id):
        cmd = ["python", "src/vla_policy/cosmos_policy_finetune.py",
               "--checkpoint", str(self.model_dir / "ppo/best_model.pt"),
               "--output", str(self.model_dir / pipeline_id), "--lora_rank", "16"]
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True)
        return result.returncode == 0

    def _step_evaluate(self, pipeline_id):
        cmd = ["python", "src/isaac_lab/evaluate_vla.py",
               "--checkpoint", str(self.model_dir / pipeline_id / "best_model.pt"), "--num_trials", "100"]
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True)
        return result.returncode == 0

    def _step_export(self, pipeline_id):
        cmd1 = ["python", "src/vla_policy/export_onnx.py",
                "--checkpoint", str(self.model_dir / pipeline_id / "best_model.pt"),
                "--output", str(self.model_dir / pipeline_id / "vla_model.onnx")]
        if subprocess.run(cmd1, cwd=self.project_root).returncode != 0:
            return False
        cmd2 = ["trtexec", f"--onnx={self.model_dir/pipeline_id/'vla_model.onnx'}",
                f"--saveEngine={self.model_dir/pipeline_id/'vla_model_fp16.plan'}", "--fp16", "--workspace=8192"]
        return subprocess.run(cmd2, cwd=self.project_root).returncode == 0

    def _step_deploy(self, pipeline_id):
        cmd = ["bash", "scripts/deploy_policy.sh", "--model", str(self.model_dir / pipeline_id), "--policy-id", pipeline_id]
        return subprocess.run(cmd, cwd=self.project_root).returncode == 0

    def _step_validate(self, pipeline_id):
        cmd = ["python", "src/follower_arm/validate_policy.py",
               "--engine", str(self.model_dir / pipeline_id / "vla_model_fp16.plan"), "--num_trials", "10"]
        result = subprocess.run(cmd, cwd=self.project_root, capture_output=True)
        return result.returncode == 0

    def _rollback(self):
        subprocess.run(["bash", "scripts/deploy_policy.sh", "--rollback"], cwd=self.project_root)

    def _log(self, msg):
        self.log.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        print(msg)
