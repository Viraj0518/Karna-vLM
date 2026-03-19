"""
Optimizer and scheduler utilities for multi-stage VLM training.

Provides per-component learning rate control and cosine scheduling
with warm restarts for multi-stage training.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def build_optimizer(
    model: nn.Module,
    base_lr: float = 1e-4,
    bridge_lr: Optional[float] = None,
    decoder_lr: Optional[float] = None,
    weight_decay: float = 0.01,
) -> torch.optim.AdamW:
    """Build AdamW optimizer with per-component learning rates.

    Args:
        model: The KarnaVLM model.
        base_lr: Default learning rate.
        bridge_lr: Learning rate for bridge (if different from base).
        decoder_lr: Learning rate for decoder (if different from base).
        weight_decay: Weight decay coefficient.

    Returns:
        Configured AdamW optimizer.
    """
    param_groups = []

    bridge_lr = bridge_lr or base_lr
    decoder_lr = decoder_lr or base_lr * 0.1  # Decoder gets lower LR by default

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        lr = base_lr
        wd = weight_decay

        # Component-specific LR
        if "bridge" in name:
            lr = bridge_lr
        elif "decoder" in name:
            lr = decoder_lr

        # No weight decay for bias and norms
        if "bias" in name or "norm" in name:
            wd = 0.0

        param_groups.append({
            "params": [param],
            "lr": lr,
            "weight_decay": wd,
            "name": name,
        })

    return torch.optim.AdamW(param_groups)


def build_cosine_schedule(
    optimizer: Optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
) -> LambdaLR:
    """Build cosine schedule with linear warmup.

    Args:
        optimizer: The optimizer.
        warmup_steps: Number of warmup steps.
        total_steps: Total training steps.
        min_lr_ratio: Minimum LR as fraction of peak.

    Returns:
        LambdaLR scheduler.
    """
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)
