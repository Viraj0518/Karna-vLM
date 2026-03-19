"""
Stage 2: Multitask Instruction Tuning.

Goal: Train the model to follow multimodal instructions across
diverse task types (VQA, captioning, OCR, grounding, etc.).

What's trained: Bridge + decoder LoRA adapters.
Data: Mixed instruction-following datasets.
Loss: Next-token prediction on responses (prompt masked).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from karna_vlm.training.trainer import VLMTrainer, TrainingConfig
from karna_vlm.training.lora import LoRAManager, LoRAConfig

logger = logging.getLogger(__name__)


@dataclass
class Stage2Config(TrainingConfig):
    """Stage 2 specific configuration."""

    output_dir: str = "outputs/stage2_multitask"
    num_epochs: int = 3
    learning_rate: float = 2e-5  # Lower LR for instruction tuning
    warmup_steps: int = 500
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    lora_r: int = 16
    lora_alpha: int = 32


def run_stage2(
    model: object,
    train_loader: object,
    eval_loader: Optional[object] = None,
    config: Optional[Stage2Config] = None,
) -> dict:
    """Run Stage 2: Multitask Instruction Tuning.

    Trains bridge + decoder LoRA on diverse instruction-following data.

    Args:
        model: KarnaVLM instance (with Stage 1 bridge weights loaded).
        train_loader: DataLoader with instruction-following data.
        eval_loader: Optional evaluation DataLoader.
        config: Stage 2 configuration.

    Returns:
        Training metrics dict.
    """
    config = config or Stage2Config()

    logger.info("=" * 60)
    logger.info("STAGE 2: Multitask Instruction Tuning")
    logger.info("=" * 60)

    # Freeze vision encoder
    model.vision_encoder.freeze()

    # Keep bridge trainable
    for param in model.bridge.parameters():
        param.requires_grad = True

    # Apply LoRA to decoder
    lora_mgr = LoRAManager(model)
    lora_mgr.apply_decoder_lora(
        LoRAConfig(r=config.lora_r, alpha=config.lora_alpha)
    )

    summary = lora_mgr.get_trainable_summary()
    logger.info(
        "Stage 2 trainable: bridge=%dM, decoder=%dM",
        summary["bridge"] // 1_000_000,
        summary["decoder"] // 1_000_000,
    )

    trainer = VLMTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        eval_loader=eval_loader,
    )
    return trainer.train()
