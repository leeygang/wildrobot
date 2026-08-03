from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from policy_contract.spec import PolicyBundle


MANIFEST_NAME = "bundle_manifest.json"


@dataclass(frozen=True)
class DeploymentBundle:
    root: Path
    manifest: dict[str, Any]

    @classmethod
    def load(cls, root: str | Path) -> "DeploymentBundle":
        root = Path(root)
        manifest_path = root / MANIFEST_NAME
        if not manifest_path.exists():
            raise FileNotFoundError(f"Deployment bundle manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict):
            raise ValueError(f"{MANIFEST_NAME} must contain a JSON object")
        if int(manifest.get("schema_version", 0)) != 1:
            raise ValueError(
                f"Unsupported deployment bundle schema_version: "
                f"{manifest.get('schema_version')!r}"
            )
        policies = manifest.get("policies")
        if not isinstance(policies, dict):
            raise ValueError(f"{MANIFEST_NAME} missing policies object")
        for role in ("standing", "walking"):
            entry = policies.get(role)
            if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
                raise ValueError(f"{MANIFEST_NAME} missing policies.{role}.path")
            policy_dir = cls._resolve_inside(root, entry["path"])
            PolicyBundle.load(policy_dir)
        bundle = cls(root=root, manifest=manifest)
        for shared_path in (
            bundle.mjcf_path,
            bundle.robot_config_path,
        ):
            if not shared_path.is_file():
                raise FileNotFoundError(f"Deployment bundle artifact not found: {shared_path}")
        return bundle

    @staticmethod
    def _resolve_inside(root: Path, relative_path: str) -> Path:
        root_resolved = root.resolve()
        path = (root / relative_path).resolve()
        if path != root_resolved and root_resolved not in path.parents:
            raise ValueError(f"Bundle path escapes deployment root: {relative_path!r}")
        return path

    def policy_dir(self, role: str) -> Path:
        policies = self.manifest["policies"]
        if role not in policies:
            raise ValueError(f"Deployment bundle has no policy role {role!r}")
        return self._resolve_inside(self.root, str(policies[role]["path"]))

    def policy_bundle(self, role: str) -> PolicyBundle:
        return PolicyBundle.load(self.policy_dir(role))

    def _shared_path(self, key: str) -> Path:
        shared = self.manifest.get("shared")
        if not isinstance(shared, dict) or not isinstance(shared.get(key), str):
            raise ValueError(f"{MANIFEST_NAME} missing shared.{key}")
        return self._resolve_inside(self.root, str(shared[key]))

    @property
    def mjcf_path(self) -> Path:
        return self._shared_path("mjcf")

    @property
    def robot_config_path(self) -> Path:
        return self._shared_path("robot_config")


def is_deployment_bundle(path: str | Path) -> bool:
    return (Path(path) / MANIFEST_NAME).is_file()


def resolve_policy_dir(path: str | Path, *, role: str) -> Path:
    path = Path(path)
    if is_deployment_bundle(path):
        return DeploymentBundle.load(path).policy_dir(role)
    return path
