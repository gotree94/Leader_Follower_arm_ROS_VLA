"""
Policy Registry — VLA 정책 버전 관리
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Optional


class VLApolicyRegistry:
    def __init__(self, registry_path: str = "models/policy_registry.json"):
        self.registry_path = Path(registry_path)
        self.registry = self._load()

    def _load(self) -> dict:
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {"active": None, "backup": None, "staged": [], "archived": [], "history": []}

    def save(self):
        with open(self.registry_path, "w") as f:
            json.dump(self.registry, f, indent=2)

    def register(self, policy_id: str, model_path: str, metadata: dict):
        entry = {"policy_id": policy_id, "model_path": str(model_path),
                 "created": datetime.now().isoformat(), "metadata": metadata, "status": "staged"}
        self.registry["staged"].append(entry)
        self.registry["history"].append(entry)
        self.save()

    def activate(self, policy_id: str) -> bool:
        staged = [p for p in self.registry["staged"] if p["policy_id"] == policy_id]
        if not staged:
            return False
        if self.registry["active"]:
            self.registry["backup"] = self.registry["active"]
            self.registry["backup"]["status"] = "backup"
        staged[0]["status"] = "active"
        self.registry["active"] = staged[0]
        self.registry["staged"] = [p for p in self.registry["staged"] if p["policy_id"] != policy_id]
        self._update_symlink(policy_id)
        self.save()
        return True

    def rollback(self) -> bool:
        if not self.registry["backup"]:
            return False
        if self.registry["active"]:
            self.registry["active"]["status"] = "archived"
            self.registry["archived"].append(self.registry["active"])
        self.registry["backup"]["status"] = "active"
        self.registry["active"] = self.registry["backup"]
        self.registry["backup"] = None
        self._update_symlink(self.registry["active"]["policy_id"])
        self.save()
        return True

    def _update_symlink(self, policy_id: str):
        active = Path("models/vla/active")
        target = Path(f"models/vla/{policy_id}")
        if active.exists() or active.is_symlink():
            active.unlink()
        os.symlink(target, active)

    def get_active(self) -> Optional[dict]:
        return self.registry.get("active")

    def get_backup(self) -> Optional[dict]:
        return self.registry.get("backup")
