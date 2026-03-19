"""
Dataset manifest management.

Provides tools for loading, validating, and registering dataset
manifests for governance and provenance tracking.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import yaml

from karna_vlm.data.schemas import DatasetManifest, License, TaskType

logger = logging.getLogger(__name__)

# Global manifest registry
_MANIFEST_REGISTRY: dict[str, DatasetManifest] = {}


def register_manifest(manifest: DatasetManifest) -> None:
    """Register a dataset manifest in the global registry."""
    _MANIFEST_REGISTRY[manifest.name] = manifest
    logger.info("Registered manifest: %s (v%s, %s)", manifest.name, manifest.version, manifest.license.value)


def get_manifest(name: str) -> Optional[DatasetManifest]:
    """Retrieve a registered manifest by name."""
    return _MANIFEST_REGISTRY.get(name)


def list_manifests() -> list[str]:
    """List all registered manifest names."""
    return list(_MANIFEST_REGISTRY.keys())


def load_manifest(path: str | Path) -> DatasetManifest:
    """Load a manifest from a YAML or JSON file.

    Args:
        path: Path to manifest file.

    Returns:
        DatasetManifest instance.
    """
    path = Path(path)
    with open(path, "r") as f:
        if path.suffix in (".yaml", ".yml"):
            data = yaml.safe_load(f)
        else:
            data = json.load(f)

    # Convert string enums
    if "license" in data:
        data["license"] = License(data["license"])
    if "task_types" in data:
        data["task_types"] = [TaskType(t) for t in data["task_types"]]

    manifest = DatasetManifest(**{
        k: v for k, v in data.items()
        if k in DatasetManifest.__dataclass_fields__
    })

    register_manifest(manifest)
    return manifest


def save_manifest(manifest: DatasetManifest, path: str | Path) -> None:
    """Save a manifest to YAML.

    Args:
        manifest: Manifest to save.
        path: Output file path.
    """
    from dataclasses import asdict
    data = asdict(manifest)
    data["license"] = manifest.license.value
    data["task_types"] = [t.value for t in manifest.task_types]

    path = Path(path)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def validate_commercial_safety(manifest_names: list[str]) -> dict[str, bool]:
    """Check which datasets are safe for commercial use.

    Args:
        manifest_names: List of dataset names to check.

    Returns:
        Dict mapping dataset name -> commercial safety boolean.
    """
    results = {}
    for name in manifest_names:
        m = get_manifest(name)
        if m is None:
            results[name] = False
            logger.warning("No manifest found for '%s' — marking as unsafe", name)
        else:
            results[name] = m.is_commercial_safe()
    return results
