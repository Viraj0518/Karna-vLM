"""
Core VLM trainer.

Handles the training loop with support for:
- Mixed precision training
- Gradient accumulation
- Warm-up scheduling
- Multi-stage training orchestration
- Logging and checkpointing
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training hyperparameters.

    Attributes:
        output_dir: Directory for checkpoints and logs.
        num_epochs: Number of training epochs.
        max_steps: Maximum training steps (overrides epochs if set).
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Steps to accumulate before optimizer step.
        learning_rate: Peak learning rate.
        weight_decay: Weight decay coefficient.
        warmup_steps: Number of warmup steps.
        max_grad_norm: Gradient clipping norm.
        fp16: Use FP16 mixed precision.
        bf16: Use BF16 mixed precision.
        logging_steps: Log every N steps.
        save_steps: Save checkpoint every N steps.
        eval_steps: Evaluate every N steps.
        seed: Random seed.
    """

    output_dir: str = "outputs"
    num_epochs: int = 3
    max_steps: int = -1
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = True
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    seed: int = 42


class VLMTrainer:
    """Training loop for Karna VLM.

    Manages the full training pipeline including optimizer setup,
    scheduling, mixed precision, and checkpointing.

    Args:
        model: The KarnaVLM model.
        config: Training configuration.
        train_loader: Training data loader.
        eval_loader: Optional evaluation data loader.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Setup optimizer (only trainable params)
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # Mixed precision
        self.scaler = None
        self.amp_dtype = torch.float32
        if config.bf16 and torch.cuda.is_bf16_supported():
            self.amp_dtype = torch.bfloat16
        elif config.fp16:
            self.amp_dtype = torch.float16
            self.scaler = torch.amp.GradScaler("cuda")

        # State
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")

        # Output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build AdamW optimizer with weight decay separation."""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bias" in name or "norm" in name or "layernorm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        return torch.optim.AdamW(param_groups, lr=self.config.learning_rate)

    def _build_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """Build linear warmup + cosine decay scheduler."""
        warmup = self.config.warmup_steps

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            return 1.0  # constant after warmup (extend to cosine if needed)

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(self) -> dict[str, float]:
        """Run the full training loop.

        Returns:
            Dict with final training metrics.
        """
        logger.info("Starting training: %d epochs, lr=%.2e", self.config.num_epochs, self.config.learning_rate)

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        logger.info("Trainable params: %dM / %dM total (%.1f%%)", trainable // 1e6, total // 1e6, 100 * trainable / total)

        total_loss = 0.0
        num_steps = 0
        start_time = time.time()

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = self._train_epoch()
            total_loss += epoch_loss
            num_steps += 1

            # Eval
            if self.eval_loader is not None:
                eval_loss = self._evaluate()
                logger.info("Epoch %d eval loss: %.4f", epoch, eval_loss)

                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self._save_checkpoint("best")

            if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                break

        elapsed = time.time() - start_time
        logger.info("Training complete in %.1f minutes", elapsed / 60)

        return {
            "train_loss": total_loss / max(num_steps, 1),
            "best_eval_loss": self.best_eval_loss,
            "total_steps": self.global_step,
            "elapsed_seconds": elapsed,
        }

    def _train_epoch(self) -> float:
        """Run one training epoch."""
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            # Forward pass with AMP
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=(self.amp_dtype != torch.float32)):
                output = self.model(
                    inputs_embeds=batch.get("inputs_embeds"),
                    attention_mask=batch.get("attention_mask"),
                    labels=batch.get("labels"),
                )
                loss = output.loss / self.config.gradient_accumulation_steps

            # Backward
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            epoch_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            # Optimizer step
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.max_grad_norm
                    )
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    avg_loss = epoch_loss / num_batches
                    logger.info(
                        "Step %d | Loss %.4f | LR %.2e",
                        self.global_step, avg_loss, lr,
                    )

                # Checkpointing
                if self.global_step % self.config.save_steps == 0:
                    self._save_checkpoint(f"step_{self.global_step}")

                if self.config.max_steps > 0 and self.global_step >= self.config.max_steps:
                    break

        return epoch_loss / max(num_batches, 1)

    @torch.no_grad()
    def _evaluate(self) -> float:
        """Run evaluation loop."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in self.eval_loader:
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            with torch.amp.autocast("cuda", dtype=self.amp_dtype, enabled=(self.amp_dtype != torch.float32)):
                output = self.model(
                    inputs_embeds=batch.get("inputs_embeds"),
                    attention_mask=batch.get("attention_mask"),
                    labels=batch.get("labels"),
                )
            total_loss += output.loss.item()
            num_batches += 1

        return total_loss / max(num_batches, 1)

    def _save_checkpoint(self, name: str) -> None:
        """Save a training checkpoint."""
        ckpt_dir = self.output_dir / name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save bridge weights
        torch.save(
            self.model.bridge.state_dict(),
            ckpt_dir / "bridge_weights.pt",
        )

        # Save optimizer state
        torch.save(
            {
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "global_step": self.global_step,
                "epoch": self.epoch,
                "best_eval_loss": self.best_eval_loss,
            },
            ckpt_dir / "training_state.pt",
        )

        # Save config
        if hasattr(self.model, "config"):
            self.model.config.to_yaml(ckpt_dir / "config.yaml")

        logger.info("Checkpoint saved: %s", ckpt_dir)
