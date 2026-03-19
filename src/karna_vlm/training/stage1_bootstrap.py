"""
Stage 1: Bootstrap Alignment Training.

Goal: Align the vision encoder's feature space with the decoder's token
space through the bridge. Uses simple captioning data.

What's trained: Bridge only (vision encoder frozen, decoder frozen).
Data: Image-caption pairs (CC3M, SBU, LAION subset, etc.).
Loss: Standard next-token prediction on captions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from karna_vlm.training.trainer import VLMTrainer, TrainingConfig

logger = logging.getLogger(__name__)


@dataclass
class Stage1Config(TrainingConfig):
    """Stage 1 specific configuration."""

    output_dir: str = "outputs/stage1_bootstrap"
    num_epochs: int = 1
    learning_rate: float = 1e-3  # Higher LR for bridge-only training
    warmup_steps: int = 200
    batch_size: int = 16
    gradient_accumulation_steps: int = 2


def run_stage1(
    model: object,
    train_loader: object,
    eval_loader: Optional[object] = None,
    config: Optional[Stage1Config] = None,
) -> dict:
    """Run Stage 1: Bootstrap Alignment.

    This stage trains ONLY the bridge module on image-caption pairs.
    Vision encoder and decoder are completely frozen.

    Args:
        model: KarnaVLM instance.
        train_loader: DataLoader with captioning data.
        eval_loader: Optional evaluation DataLoader.
        config: Stage 1 configuration.

    Returns:
        Training metrics dict.
    """
    config = config or Stage1Config()

    logger.info("=" * 60)
    logger.info("STAGE 1: Bootstrap Alignment")
    logger.info("=" * 60)

    # Ensure correct freezing
    model.vision_encoder.freeze()
    model.decoder.freeze()

    # Unfreeze bridge
    for param in model.bridge.parameters():
        param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Stage 1 trainable params: %dM (bridge only)", trainable // 1_000_000)

    trainer = VLMTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        eval_loader=eval_loader,
    )
    return trainer.train()
