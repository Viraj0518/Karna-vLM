"""
Stage 3: Domain Specialization.

Goal: Fine-tune the model for a specific domain (medical, legal,
finance, OCR, etc.) using domain-specific data and adapters.

What's trained: Bridge LoRA + decoder domain adapter.
Data: Domain-specific datasets.
Output: Domain pack (bridge weights + adapter + config).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import yaml

from karna_vlm.training.trainer import VLMTrainer, TrainingConfig
from karna_vlm.training.lora import LoRAManager, LoRAConfig

logger = logging.getLogger(__name__)


@dataclass
class Stage3Config(TrainingConfig):
    """Stage 3 specific configuration."""

    output_dir: str = "outputs/stage3_domain"
    domain_name: str = "general"
    num_epochs: int = 5
    learning_rate: float = 1e-5
    warmup_steps: int = 100
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    lora_r: int = 8
    lora_alpha: int = 16


def run_stage3(
    model: object,
    train_loader: object,
    eval_loader: Optional[object] = None,
    config: Optional[Stage3Config] = None,
) -> dict:
    """Run Stage 3: Domain Specialization.

    Fine-tunes for a specific domain, producing a domain pack.

    Args:
        model: KarnaVLM instance (with Stage 2 weights).
        train_loader: DataLoader with domain-specific data.
        eval_loader: Optional evaluation DataLoader.
        config: Stage 3 configuration.

    Returns:
        Training metrics dict.
    """
    config = config or Stage3Config()

    logger.info("=" * 60)
    logger.info("STAGE 3: Domain Specialization — %s", config.domain_name)
    logger.info("=" * 60)

    # Freeze vision encoder
    model.vision_encoder.freeze()

    # Apply bridge LoRA (lighter than full bridge training)
    lora_mgr = LoRAManager(model)
    lora_mgr.apply_bridge_lora(
        LoRAConfig(r=config.lora_r, alpha=config.lora_alpha),
        adapter_name=f"bridge_{config.domain_name}",
    )

    # Apply decoder domain adapter
    lora_mgr.apply_decoder_lora(
        LoRAConfig(r=config.lora_r, alpha=config.lora_alpha),
        adapter_name=f"decoder_{config.domain_name}",
    )

    trainer = VLMTrainer(
        model=model,
        config=config,
        train_loader=train_loader,
        eval_loader=eval_loader,
    )
    metrics = trainer.train()

    # Save domain pack
    _save_domain_pack(model, config, metrics)

    return metrics


def _save_domain_pack(model: object, config: Stage3Config, metrics: dict) -> None:
    """Save a domain adaptation pack."""
    pack_dir = Path(config.output_dir) / f"domain_pack_{config.domain_name}"
    pack_dir.mkdir(parents=True, exist_ok=True)

    # Save bridge weights
    torch.save(model.bridge.state_dict(), pack_dir / "bridge_weights.pt")

    # Save domain config
    domain_meta = {
        "domain_name": config.domain_name,
        "base_model": getattr(model.config, "model_name", "unknown"),
        "training_metrics": {k: float(v) if isinstance(v, (int, float)) else str(v) for k, v in metrics.items()},
        "lora_r": config.lora_r,
        "lora_alpha": config.lora_alpha,
    }
    with open(pack_dir / "config.yaml", "w") as f:
        yaml.dump(domain_meta, f, default_flow_style=False)

    logger.info("Domain pack saved: %s", pack_dir)
