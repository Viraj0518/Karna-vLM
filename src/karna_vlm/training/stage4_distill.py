"""
Stage 4: Distillation & Compression.

Goal: Compress the trained model for efficient deployment through
knowledge distillation, quantization-aware training, or pruning.

This stage produces deployment-ready model variants.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from karna_vlm.training.trainer import TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class Stage4Config(TrainingConfig):
    """Stage 4 specific configuration."""

    output_dir: str = "outputs/stage4_distill"
    teacher_model_path: str = ""
    distill_temperature: float = 2.0
    distill_alpha: float = 0.5  # weight of distillation loss vs task loss
    num_epochs: int = 2
    learning_rate: float = 5e-6


class DistillationLoss(nn.Module):
    """Knowledge distillation loss combining KL-div and task loss.

    L = alpha * KL(teacher || student) + (1 - alpha) * CE(labels, student)

    Args:
        temperature: Softmax temperature for distillation.
        alpha: Weight of distillation loss vs task loss.
    """

    def __init__(self, temperature: float = 2.0, alpha: float = 0.5) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            student_logits: [B, seq_len, vocab_size]
            teacher_logits: [B, seq_len, vocab_size]
            labels: Optional [B, seq_len] for task loss.

        Returns:
            Combined distillation + task loss.
        """
        T = self.temperature

        # KL divergence on softened distributions
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (T ** 2)

        if labels is not None and self.alpha < 1.0:
            # Standard cross-entropy task loss
            task_loss = F.cross_entropy(
                student_logits.view(-1, student_logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            return self.alpha * kl_loss + (1 - self.alpha) * task_loss

        return kl_loss


def quantize_bridge(
    bridge: nn.Module,
    bits: int = 8,
    method: str = "dynamic",
) -> nn.Module:
    """Apply quantization to bridge module.

    Args:
        bridge: The bridge module to quantize.
        bits: Quantization bit width (8 or 4).
        method: Quantization method ('dynamic', 'static', 'qat').

    Returns:
        Quantized bridge module.
    """
    if method == "dynamic" and bits == 8:
        quantized = torch.ao.quantization.quantize_dynamic(
            bridge,
            {nn.Linear},
            dtype=torch.qint8,
        )
        logger.info("Applied dynamic INT8 quantization to bridge")
        return quantized

    logger.warning("Quantization method '%s' with %d bits not yet implemented", method, bits)
    return bridge


def run_stage4(
    student_model: object,
    teacher_model: Optional[object] = None,
    train_loader: Optional[object] = None,
    config: Optional[Stage4Config] = None,
) -> dict:
    """Run Stage 4: Distillation & Compression.

    If a teacher model is provided, performs knowledge distillation.
    Otherwise, applies quantization and compression directly.

    Args:
        student_model: The model to compress.
        teacher_model: Optional larger teacher model.
        train_loader: Training data loader.
        config: Stage 4 configuration.

    Returns:
        Metrics dict with compression stats.
    """
    config = config or Stage4Config()

    logger.info("=" * 60)
    logger.info("STAGE 4: Distillation & Compression")
    logger.info("=" * 60)

    metrics: dict = {}

    # Report initial sizes
    total_params = sum(p.numel() for p in student_model.parameters())
    logger.info("Student model params: %dM", total_params // 1_000_000)

    # Quantize bridge
    if hasattr(student_model, "bridge"):
        bridge_params_before = sum(p.numel() for p in student_model.bridge.parameters())
        student_model.bridge = quantize_bridge(student_model.bridge)
        metrics["bridge_params_before"] = bridge_params_before

    metrics["total_params"] = total_params
    return metrics
