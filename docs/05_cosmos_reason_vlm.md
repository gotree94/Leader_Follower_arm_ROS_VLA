# Cosmos Reason — Vision-Language Model for Task Planning

> NVIDIA Cosmos Reason VLM을 이용한 자연어 명령 → 로봇 태스크 플랜 변환

---

## 1. Cosmos Reason 개요

Cosmos Reason은 NVIDIA의 VLM(Vision-Language Model)으로, 입력 이미지와 자연어 명령을 분석하여 구조화된 태스크 플랜을 출력합니다.

### 1.1 역할

```
사용자: "파란 블록을 집어서 빨간 상자 위에 쌓아줘"
         │
         ▼
┌─────────────────────────────────────────────┐
│              Cosmos Reason VLM              │
│                                             │
│  1. Scene Understanding (객체 탐지 + 위치)   │
│  2. Task Decomposition (단계 분할)           │
│  3. Spatial Reasoning (공간 추론)            │
│  4. Constraints Extraction (제약 조건 추출)  │
│                                             │
└───────────────────┬─────────────────────────┘
                    │
                    ▼
{
  "task": "stack_blocks",
  "steps": [
    {"action": "approach", "target": "blue_block", ...},
    {"action": "grasp", "target": "blue_block", ...},
    {"action": "move", "target": "red_box_top", ...},
    {"action": "release", "target": "blue_block", ...}
  ]
}
```

### 1.2 아키텍처

```
Cosmos Reason Architecture:
┌────────────────────────────────────────┐
│  Vision Encoder (ViT-L/14)             │ ← RGB image input
│      │                                 │
│  Projection Layer                      │
│      │                                 │
│  LLM Backbone (Nemotron-4 340B)        │ ← text instruction + image tokens
│      │                                 │
│  Instruction Tuned (RLHF)              │
│      │                                 │
│  Structured Output Decoder             │ ← JSON task plan
└────────────────────────────────────────┘
```

### 1.3 버전 2.0 주요 기능

| 기능 | 설명 |
|------|------|
| **6D Pose Estimation** | 객체의 3D 위치와 회전 추정 (카메라 기준) |
| **Spatial Reasoning** | "왼쪽에 있는", "오른쪽 상단" 등 공간 관계 이해 |
| **Action Sequencing** | 복잡한 태스크를 순차적 액션으로 분해 |
| **Failure Explanation** | 태스크 실패 시 원인 분석 |
| **Few-shot Learning** | 몇 가지 예시만으로 새로운 태스크 이해 |
| **Multi-turn Dialogue** | 연속적인 명령 이해 ("이제 그걸 옮겨줘") |

---

## 2. API Integration

### 2.1 Python Client

```python
# src/cosmos_reason/task_planner.py
"""
Cosmos Reason API 클라이언트 — 자연어 명령 → 태스크 플랜
"""
import os
import json
import base64
import requests
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class ObjectInfo:
    """탐지된 객체 정보"""
    id: str
    name: str
    class_name: str
    pose: List[float]  # [x, y, z, qw, qx, qy, qz]
    color: List[float]  # [R, G, B]
    dimensions: List[float]  # [width, height, depth]
    confidence: float


@dataclass
class TaskStep:
    """태스크 단계"""
    id: int
    action: str  # approach, grasp, lift, move, place, release, push, pull
    target: str
    target_pose: Optional[List[float]] = None
    gripper_width: Optional[float] = None
    description: str = ""
    constraints: Dict = field(default_factory=dict)


@dataclass
class TaskPlan:
    """전체 태스크 플랜"""
    task: str
    instruction: str
    objects: List[ObjectInfo]
    steps: List[TaskStep]
    success_condition: str
    workspace_bounds: Optional[Dict] = None


class CosmosReasonClient:
    """
    Cosmos Reason VLM 클라이언트
    
    사용법:
        client = CosmosReasonClient(api_key="...")
        plan = client.plan_task(
            instruction="Pick up the blue block",
            rgb_image=rgb_array,
            depth_image=depth_array
        )
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        model: str = "cosmos-reason-v2"
    ):
        self.api_key = api_key or os.getenv("COSMOS_API_KEY")
        self.endpoint = endpoint or os.getenv(
            "COSMOS_API_ENDPOINT",
            "https://api.nvidia.com/v1/cosmos"
        )
        self.model = model
        
        if not self.api_key:
            raise ValueError(
                "COSMOS_API_KEY not set. "
                "export COSMOS_API_KEY='your-key'"
            )
    
    def plan_task(
        self,
        instruction: str,
        rgb_image: np.ndarray,
        depth_image: Optional[np.ndarray] = None,
        context: Optional[str] = None,
        temperature: float = 0.1,
        max_retries: int = 3
    ) -> TaskPlan:
        """
        자연어 명령 → 태스크 플랜 변환
        
        Args:
            instruction: 자연어 명령 (예: "Pick the blue block")
            rgb_image: RGB 이미지 (H, W, 3), uint8
            depth_image: Depth 이미지 (H, W, 1), float32 (옵션)
            context: 추가 컨텍스트 (예: workspace bounds)
            temperature: LLM temperature (낮을수록 결정적)
            max_retries: 재시도 횟수
            
        Returns:
            TaskPlan: 구조화된 태스크 플랜
        """
        # 이미지 → base64 인코딩
        rgb_b64 = self._encode_image(rgb_image)
        depth_b64 = self._encode_image(depth_image) if depth_image is not None else None
        
        # 시스템 프롬프트
        system_prompt = self._build_system_prompt()
        
        # 사용자 메시지
        user_message = self._build_user_message(
            instruction, rgb_b64, depth_b64, context
        )
        
        # API 호출
        for attempt in range(max_retries):
            try:
                response = self._call_api(system_prompt, user_message, temperature)
                plan = self._parse_response(response)
                return plan
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2 ** attempt)  # exponential backoff
                    continue
                raise RuntimeError(f"Cosmos Reason failed after {max_retries} attempts: {e}")
    
    def _build_system_prompt(self) -> str:
        """시스템 프롬프트 구성"""
        return """You are a robotic task planner for a dual-arm manipulation system.
You receive an RGB image of the workspace and a natural language instruction.
Your job is to:

1. Detect all relevant objects in the scene (name, position, color, size)
2. Decompose the instruction into a sequence of robotic actions
3. For each action, specify the target object, target pose, and gripper state
4. Define the success condition

Available actions:
- approach: Move end-effector to pre-grasp position (5cm above target)
- grasp: Close gripper to grasp the target
- lift: Lift the grasped object 10cm
- move: Move the end-effector to a target position
- place: Lower the object to the target surface
- release: Open gripper to release the object
- push: Apply force to move an object
- pull: Pull an object toward the end-effector
- insert: Insert one object into another
- screw: Rotate wrist while maintaining grip

Output format: JSON only, no explanations.
{
  "task": "task_type",
  "instruction": "original instruction",
  "objects": [
    {
      "id": "obj_001",
      "name": "blue_block",
      "class_name": "block",
      "pose": [x, y, z, qw, qx, qy, qz],
      "color": [R, G, B],
      "dimensions": [w, h, d],
      "confidence": 0.95
    }
  ],
  "steps": [
    {
      "id": 0,
      "action": "approach",
      "target": "blue_block",
      "target_pose": [0.3, -0.2, 0.15, 1.0, 0, 0, 0],
      "gripper_width": 0.08,
      "description": "Move above blue block"
    }
  ],
  "success_condition": "blue_block is placed on red_box",
  "workspace_bounds": {
    "x": [-0.4, 0.4],
    "y": [-0.3, 0.3],
    "z": [0.0, 0.8]
  }
}"""
    
    def _build_user_message(
        self,
        instruction: str,
        rgb_b64: str,
        depth_b64: Optional[str],
        context: Optional[str]
    ) -> Dict:
        """사용자 메시지 구성"""
        content = [
            {
                "type": "text",
                "text": f"Instruction: {instruction}"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{rgb_b64}"
                }
            }
        ]
        
        if depth_b64:
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{depth_b64}"
                }
            })
        
        if context:
            content.append({
                "type": "text",
                "text": f"Context: {context}"
            })
        
        return {"role": "user", "content": content}
    
    def _call_api(self, system_prompt: str, user_message: Dict, temperature: float) -> Dict:
        """Cosmos Reason API 호출"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                user_message
            ],
            "temperature": temperature,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"}
        }
        
        response = requests.post(
            f"{self.endpoint}/reason",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def _parse_response(self, response: Dict) -> TaskPlan:
        """API 응답 → TaskPlan 파싱"""
        content = response["choices"][0]["message"]["content"]
        data = json.loads(content)
        
        objects = [
            ObjectInfo(
                id=obj["id"],
                name=obj["name"],
                class_name=obj.get("class_name", "unknown"),
                pose=obj["pose"],
                color=obj.get("color", [0.5, 0.5, 0.5]),
                dimensions=obj.get("dimensions", [0.05, 0.05, 0.05]),
                confidence=obj.get("confidence", 1.0)
            )
            for obj in data.get("objects", [])
        ]
        
        steps = [
            TaskStep(
                id=step["id"],
                action=step["action"],
                target=step["target"],
                target_pose=step.get("target_pose"),
                gripper_width=step.get("gripper_width"),
                description=step.get("description", ""),
                constraints=step.get("constraints", {})
            )
            for step in data.get("steps", [])
        ]
        
        return TaskPlan(
            task=data.get("task", "unknown"),
            instruction=data.get("instruction", ""),
            objects=objects,
            steps=steps,
            success_condition=data.get("success_condition", ""),
            workspace_bounds=data.get("workspace_bounds")
        )
    
    def _encode_image(self, image: np.ndarray) -> str:
        """numpy array → base64 PNG"""
        import cv2
        success, buffer = cv2.imencode('.png', image)
        if not success:
            raise ValueError("Failed to encode image")
        return base64.b64encode(buffer).decode('utf-8')
```

### 2.2 ROS2 태스크 플래너 노드

```python
# src/cosmos_reason/task_planner_node.py
"""
ROS2 노드 — Cosmos Reason Task Planner
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import numpy as np

from vla_interfaces.msg import TaskPlan
from task_planner import CosmosReasonClient


class TaskPlannerNode(Node):
    """
    ROS2 Cosmos Reason Task Planner Node
    
    Subscribers:
    - /camera/rgb/image_raw: RGB image
    - /cosmos/reason/instruction: Natural language instruction
    
    Publishers:
    - /cosmos/reason/task_plan: Structured task plan
    """
    
    def __init__(self):
        super().__init__('task_planner_node')
        
        # Cosmos Reason client
        self.client = CosmosReasonClient()
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # Latest RGB image
        self.latest_rgb = None
        
        # Subscribers
        self.rgb_sub = self.create_subscription(
            Image,
            '/camera/rgb/image_raw',
            self.rgb_callback,
            10
        )
        
        self.instruction_sub = self.create_subscription(
            String,
            '/cosmos/reason/instruction',
            self.instruction_callback,
            10
        )
        
        # Publisher
        self.plan_pub = self.create_publisher(
            TaskPlan,
            '/cosmos/reason/task_plan',
            10
        )
        
        self.get_logger().info("Task Planner Node ready")
    
    def rgb_callback(self, msg: Image):
        """RGB 이미지 저장"""
        self.latest_rgb = self.bridge.imgmsg_to_cv2(msg, "bgr8")
    
    def instruction_callback(self, msg: String):
        """자연어 명령 수신 → 태스크 플랜 생성"""
        if self.latest_rgb is None:
            self.get_logger().warn("No RGB image available yet")
            return
        
        instruction = msg.data
        self.get_logger().info(f"Planning task: '{instruction}'")
        
        try:
            # Cosmos Reason 호출
            plan = self.client.plan_task(
                instruction=instruction,
                rgb_image=self.latest_rgb,
            )
            
            # ROS2 메시지 변환
            plan_msg = TaskPlan()
            plan_msg.task_id = plan.task
            plan_msg.instruction = plan.instruction
            plan_msg.action_sequence = [step.action for step in plan.steps]
            
            # Flatten poses
            flat_poses = []
            for step in plan.steps:
                if step.target_pose:
                    flat_poses.extend(step.target_pose)
            plan_msg.target_poses = flat_poses
            
            plan_msg.object_names = [obj.name for obj in plan.objects]
            
            # Publish
            self.plan_pub.publish(plan_msg)
            self.get_logger().info(f"Published task plan: {len(plan.steps)} steps")
            
        except Exception as e:
            self.get_logger().error(f"Task planning failed: {e}")
```

---

## 3. Scene Graph

### 3.1 장면 그래프

```python
# src/cosmos_reason/scene_graph.py
"""
장면 그래프 — 객체 관계 및 상태 추적
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


@dataclass
class SceneNode:
    """장면 그래프의 노드 (객체)"""
    id: str
    name: str
    class_name: str
    pose: np.ndarray  # (7,) [x,y,z,qw,qx,qy,qz]
    color: Tuple[float, float, float]
    dimensions: Tuple[float, float, float]
    confidence: float
    parent_id: Optional[str] = None
    properties: Dict = field(default_factory=dict)


@dataclass
class SceneRelation:
    """객체 간 관계"""
    source_id: str
    target_id: str
    relation: str  # "on_top_of", "left_of", "inside", "next_to", "grasped_by"
    confidence: float


class SceneGraph:
    """
    로봇 작업 공간의 장면 그래프
    
    객체 탐지 결과를 그래프 구조로 유지하여
    공간 관계 추론 및 상태 변화 추적
    """
    
    def __init__(self):
        self.nodes: Dict[str, SceneNode] = {}
        self.relations: List[SceneRelation] = []
        self.root_id = "workspace"
    
    def update_from_detection(self, objects: List[SceneNode]):
        """객체 탐지 결과로 그래프 업데이트"""
        # 기존 노드 업데이트 또는 추가
        for obj in objects:
            if obj.id in self.nodes:
                # 기존 객체 업데이트
                self.nodes[obj.id].pose = obj.pose
                self.nodes[obj.id].confidence = obj.confidence
            else:
                # 새 객체 추가
                self.nodes[obj.id] = obj
        
        # 관계 재계산
        self._compute_relations()
    
    def _compute_relations(self):
        """공간 관계 계산"""
        self.relations = []
        
        node_ids = list(self.nodes.keys())
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                a = self.nodes[node_ids[i]]
                b = self.nodes[node_ids[j]]
                
                pos_a = a.pose[:3]
                pos_b = b.pose[:3]
                
                # 수직 관계 (on_top_of)
                vertical_dist = abs(pos_a[2] - pos_b[2])
                horizontal_dist = np.linalg.norm(pos_a[:2] - pos_b[:2])
                
                if vertical_dist > 0.02 and horizontal_dist < 0.1:
                    if pos_a[2] > pos_b[2]:
                        self.relations.append(SceneRelation(a.id, b.id, "on_top_of", 0.9))
                    else:
                        self.relations.append(SceneRelation(b.id, a.id, "on_top_of", 0.9))
                
                # 수평 관계
                angle = np.arctan2(pos_b[1] - pos_a[1], pos_b[0] - pos_a[0])
                if horizontal_dist < 0.15:
                    if -np.pi/4 < angle < np.pi/4:
                        self.relations.append(SceneRelation(a.id, b.id, "right_of", 0.7))
                    elif 3*np.pi/4 < angle or angle < -3*np.pi/4:
                        self.relations.append(SceneRelation(a.id, b.id, "left_of", 0.7))
    
    def get_object_by_name(self, name: str) -> Optional[SceneNode]:
        """이름으로 객체 찾기 (부분 매칭)"""
        name_lower = name.lower()
        for node in self.nodes.values():
            if name_lower in node.name.lower():
                return node
            if name_lower in node.class_name.lower():
                return node
        return None
    
    def get_objects_by_class(self, class_name: str) -> List[SceneNode]:
        """클래스로 객체 찾기"""
        return [n for n in self.nodes.values() if n.class_name == class_name]
    
    def get_pose(self, object_name: str) -> Optional[np.ndarray]:
        """객체 위치 조회"""
        obj = self.get_object_by_name(object_name)
        return obj.pose if obj else None
    
    def get_relation(self, source: str, relation: str) -> List[str]:
        """특정 관계에 있는 객체 목록"""
        targets = []
        for rel in self.relations:
            if rel.source_id == source and rel.relation == relation:
                targets.append(rel.target_id)
            elif rel.target_id == source:
                inv_relations = {
                    "on_top_of": "under",
                    "left_of": "right_of",
                    "right_of": "left_of",
                }
                if rel.relation in inv_relations and inv_relations[rel.relation] == relation:
                    targets.append(rel.source_id)
        return targets
```

---

## 4. Object Detection

### 4.1 객체 탐지

```python
# src/cosmos_reason/object_detector.py
"""
객체 탐지 + 6D Pose 추정
"""
import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Detection:
    name: str
    class_name: str
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: Optional[np.ndarray] = None
    confidence: float = 1.0
    pose_2d: Optional[Tuple[float, float]] = None
    pose_6d: Optional[np.ndarray] = None  # (7,) [x,y,z,qw,qx,qy,qz]


class ObjectDetector:
    """
    객체 탐지 + 6D Pose 추정
    
    지원 모델:
    - GroundingDINO (open-vocabulary detection)
    - Detic (vocabulary-free detection)
    - FoundationPose (6D pose)
    """
    
    def __init__(
        self,
        detection_model: str = "grounding_dino",
        pose_model: str = "foundation_pose",
        confidence_threshold: float = 0.3,
        device: str = "cuda"
    ):
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        # Detection model
        if detection_model == "grounding_dino":
            self._init_grounding_dino()
        elif detection_model == "detic":
            self._init_detic()
        
        # Pose model
        if pose_model == "foundation_pose":
            self._init_foundation_pose()
    
    def _init_grounding_dino(self):
        """GroundingDINO 모델 초기화"""
        from groundingdino.util.inference import Model
        self.detector = Model(
            model_config_path="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
            model_checkpoint_path="models/groundingdino_swint_ogc.pth",
            device=self.device
        )
    
    def _init_detic(self):
        """Detic 모델 초기화"""
        # Detic class-based detection
        self.detector = None  # placeholder
    
    def _init_foundation_pose(self):
        """FoundationPose 6D pose 추정"""
        self.pose_estimator = None  # placeholder
    
    def detect(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        class_names: List[str],
        intrinsics: Optional[np.ndarray] = None
    ) -> List[Detection]:
        """
        객체 탐지 수행
        
        Args:
            rgb: (H, W, 3) uint8
            depth: (H, W) float32 meters
            class_names: 탐지할 클래스 목록
            intrinsics: (3, 3) camera matrix
            
        Returns:
            List[Detection]
        """
        # GroundingDINO detection
        detections = self._run_detection(rgb, class_names)
        
        # 6D Pose 추정
        for det in detections:
            if det.confidence > self.confidence_threshold:
                pose = self._estimate_pose(rgb, depth, det.bbox, intrinsics)
                if pose is not None:
                    det.pose_6d = pose
        
        return detections
    
    def _run_detection(self, rgb: np.ndarray, class_names: List[str]) -> List[Detection]:
        """탐지 실행"""
        # GroundingDINO expects text prompts
        text_prompt = ". ".join(class_names)
        
        boxes, logits, phrases = self.detector.predict_with_caption(
            rgb, text_prompt, box_threshold=0.3, text_threshold=0.25
        )
        
        detections = []
        for box, logit, phrase in zip(boxes, logits, phrases):
            detections.append(Detection(
                name=phrase,
                class_name=phrase,
                bbox=(
                    int(box[0]), int(box[1]),
                    int(box[2]), int(box[3])
                ),
                confidence=float(logit),
            ))
        
        return detections
    
    def _estimate_pose(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        bbox: Tuple[int, int, int, int],
        intrinsics: Optional[np.ndarray]
    ) -> Optional[np.ndarray]:
        """2D bbox → 6D pose"""
        if intrinsics is None:
            # Default RealSense D435 intrinsics
            intrinsics = np.array([
                [615.0, 0, 320.0],
                [0, 615.0, 240.0],
                [0, 0, 1.0]
            ])
        
        # Bbox center
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        
        # Depth at center
        dz = float(depth[int(cy), int(cx)])
        if np.isnan(dz) or dz <= 0:
            return None
        
        # Pixel → Camera 3D
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        ppx, ppy = intrinsics[0, 2], intrinsics[1, 2]
        
        x = (cx - ppx) * dz / fx
        y = (cy - ppy) * dz / fy
        z = dz
        
        # Assume gripper-down orientation
        return np.array([x, y, z, 1.0, 0.0, 0.0, 0.0])
```
