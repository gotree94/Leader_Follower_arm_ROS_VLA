"""
Cosmos Reason Task Planner
자연어 명령 → 태스크 플랜 변환
"""
import os
import json
import base64
import requests
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class ObjectInfo:
    id: str
    name: str
    class_name: str
    pose: List[float]
    color: List[float]
    dimensions: List[float]
    confidence: float


@dataclass
class TaskStep:
    id: int
    action: str
    target: str
    target_pose: Optional[List[float]] = None
    gripper_width: Optional[float] = None
    description: str = ""
    constraints: dict = field(default_factory=dict)


@dataclass
class TaskPlan:
    task: str
    instruction: str
    objects: List[ObjectInfo]
    steps: List[TaskStep]
    success_condition: str
    workspace_bounds: Optional[dict] = None


class CosmosReasonClient:
    def __init__(self, api_key: Optional[str] = None, endpoint: Optional[str] = None):
        self.api_key = api_key or os.getenv("COSMOS_API_KEY")
        self.endpoint = endpoint or os.getenv("COSMOS_API_ENDPOINT", "https://api.nvidia.com/v1/cosmos")
        if not self.api_key:
            raise ValueError("COSMOS_API_KEY not set")

    def plan_task(self, instruction: str, rgb_image: np.ndarray, depth_image: Optional[np.ndarray] = None,
                  context: Optional[str] = None, temperature: float = 0.1, max_retries: int = 3) -> TaskPlan:
        rgb_b64 = self._encode_image(rgb_image)
        depth_b64 = self._encode_image(depth_image) if depth_image is not None else None
        system_prompt = self._build_system_prompt()
        user_message = self._build_user_message(instruction, rgb_b64, depth_b64, context)
        for attempt in range(max_retries):
            try:
                response = self._call_api(system_prompt, user_message, temperature)
                return self._parse_response(response)
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)
                    continue
                raise RuntimeError(f"Cosmos Reason failed: {e}")

    def _build_system_prompt(self) -> str:
        return """You are a robotic task planner for a dual-arm manipulation system. 
Analyze the RGB image and natural language instruction. Output a JSON task plan with detected objects and action steps.
Available actions: approach, grasp, lift, move, place, release, push, pull, insert.
Output format: JSON with 'task', 'instruction', 'objects', 'steps', 'success_condition'."""

    def _build_user_message(self, instruction, rgb_b64, depth_b64, context):
        content = [{"type": "text", "text": f"Instruction: {instruction}"},
                   {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{rgb_b64}"}}]
        if depth_b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{depth_b64}"}})
        if context:
            content.append({"type": "text", "text": f"Context: {context}"})
        return {"role": "user", "content": content}

    def _call_api(self, system_prompt, user_message, temperature):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": "cosmos-reason-v2", "messages": [
            {"role": "system", "content": system_prompt}, user_message],
            "temperature": temperature, "max_tokens": 2048, "response_format": {"type": "json_object"}}
        response = requests.post(f"{self.endpoint}/reason", headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    def _parse_response(self, response) -> TaskPlan:
        content = response["choices"][0]["message"]["content"]
        data = json.loads(content)
        objects = [ObjectInfo(**o) for o in data.get("objects", [])]
        steps = [TaskStep(**s) for s in data.get("steps", [])]
        return TaskPlan(task=data.get("task", "unknown"), instruction=data.get("instruction", ""),
                        objects=objects, steps=steps, success_condition=data.get("success_condition", ""))

    def _encode_image(self, image: np.ndarray) -> str:
        import cv2
        success, buffer = cv2.imencode('.png', image)
        if not success:
            raise ValueError("Failed to encode image")
        return base64.b64encode(buffer).decode('utf-8')
