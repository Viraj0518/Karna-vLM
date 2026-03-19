"""I/O utilities for loading/saving models, configs, and data."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Union

import torch
import yaml

logger = logging.getLogger(__name__)


def load_yaml(path: Union[str, Path]) -> dict[str, Any]:
    """Load a YAML file."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data: dict[str, Any], path: Union[str, Path]) -> None:
    """Save data to a YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def load_json(path: Union[str, Path]) -> Any:
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(data: Any, path: Union[str, Path], indent: int = 2) -> None:
    """Save data to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=indent, default=str)


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    """Count model parameters by component.

    Returns:
        Dict with 'total', 'trainable', and per-module counts.
    """
    result = {"total": 0, "trainable": 0}
    for name, param in model.named_parameters():
        result["total"] += param.numel()
        if param.requires_grad:
            result["trainable"] += param.numel()
            # Top-level component
            component = name.split(".")[0]
            result[f"trainable_{component}"] = result.get(f"trainable_{component}", 0) + param.numel()
    return result
