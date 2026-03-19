"""
Checkpoint management for Karna VLM.

Supports:
- Component-level checkpointing (bridge, decoder adapters separately)
- Training state save/resume
- Checkpoint pruning
- Checkpoint comparison
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import yaml

logger = logging.getLogger(__name__)


@dataclass
class CheckpointInfo:
    """Metadata about a saved checkpoint."""

    path: Path
    step: int
    epoch: int
    loss: float
    components: list[str]  # which components were saved
    timestamp: str = ""


class CheckpointManager:
    """Manages model checkpoints with component-level granularity.

    Args:
        output_dir: Base directory for checkpoints.
        max_checkpoints: Maximum number of checkpoints to keep.
        save_bridge: Whether to save bridge weights.
        save_decoder_adapter: Whether to save decoder adapter.
        save_optimizer: Whether to save optimizer state.
    """

    def __init__(
        self,
        output_dir: str | Path,
        max_checkpoints: int = 5,
        save_bridge: bool = True,
        save_decoder_adapter: bool = True,
        save_optimizer: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_bridge = save_bridge
        self.save_decoder_adapter = save_decoder_adapter
        self.save_optimizer = save_optimizer
        self.checkpoints: list[CheckpointInfo] = []

    def save(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        step: int = 0,
        epoch: int = 0,
        loss: float = 0.0,
        extra: Optional[dict] = None,
    ) -> Path:
        """Save a checkpoint.

        Args:
            model: The KarnaVLM model.
            optimizer: Optional optimizer state.
            scheduler: Optional scheduler state.
            step: Current training step.
            epoch: Current epoch.
            loss: Current loss value.
            extra: Additional metadata to save.

        Returns:
            Path to the saved checkpoint directory.
        """
        import datetime

        ckpt_name = f"checkpoint-{step}"
        ckpt_dir = self.output_dir / ckpt_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        components_saved = []

        # Save bridge weights
        if self.save_bridge and hasattr(model, "bridge"):
            torch.save(model.bridge.state_dict(), ckpt_dir / "bridge_weights.pt")
            components_saved.append("bridge")

        # Save decoder adapter (if using LoRA/PEFT)
        if self.save_decoder_adapter and hasattr(model, "decoder"):
            decoder_model = model.decoder.model
            if hasattr(decoder_model, "save_pretrained"):
                decoder_model.save_pretrained(ckpt_dir / "decoder_adapter")
                components_saved.append("decoder_adapter")

        # Save optimizer
        if self.save_optimizer and optimizer is not None:
            state = {"optimizer": optimizer.state_dict()}
            if scheduler is not None:
                state["scheduler"] = scheduler.state_dict()
            torch.save(state, ckpt_dir / "optimizer.pt")
            components_saved.append("optimizer")

        # Save metadata
        meta = {
            "step": step,
            "epoch": epoch,
            "loss": float(loss),
            "components": components_saved,
            "timestamp": datetime.datetime.now().isoformat(),
        }
        if extra:
            meta.update(extra)

        with open(ckpt_dir / "metadata.yaml", "w") as f:
            yaml.dump(meta, f, default_flow_style=False)

        # Save config
        if hasattr(model, "config"):
            model.config.to_yaml(ckpt_dir / "config.yaml")

        # Track checkpoint
        info = CheckpointInfo(
            path=ckpt_dir,
            step=step,
            epoch=epoch,
            loss=loss,
            components=components_saved,
            timestamp=meta["timestamp"],
        )
        self.checkpoints.append(info)

        # Prune old checkpoints
        self._prune()

        logger.info(
            "Checkpoint saved: %s (step=%d, loss=%.4f, components=%s)",
            ckpt_dir, step, loss, components_saved,
        )
        return ckpt_dir

    def load(
        self,
        model: nn.Module,
        checkpoint_path: str | Path,
        load_bridge: bool = True,
        load_optimizer: bool = False,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ) -> dict[str, Any]:
        """Load a checkpoint.

        Args:
            model: The KarnaVLM model to load into.
            checkpoint_path: Path to checkpoint directory.
            load_bridge: Whether to load bridge weights.
            load_optimizer: Whether to load optimizer state.
            optimizer: Optimizer to restore state into.
            scheduler: Scheduler to restore state into.

        Returns:
            Metadata dict from the checkpoint.
        """
        ckpt_dir = Path(checkpoint_path)

        # Load metadata
        meta_path = ckpt_dir / "metadata.yaml"
        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = yaml.safe_load(f)

        # Load bridge
        if load_bridge:
            bridge_path = ckpt_dir / "bridge_weights.pt"
            if bridge_path.exists():
                state = torch.load(bridge_path, map_location="cpu", weights_only=True)
                model.bridge.load_state_dict(state)
                logger.info("Loaded bridge weights from %s", bridge_path)

        # Load decoder adapter
        adapter_dir = ckpt_dir / "decoder_adapter"
        if adapter_dir.exists() and hasattr(model, "attach_adapter"):
            model.attach_adapter(adapter_dir)
            logger.info("Loaded decoder adapter from %s", adapter_dir)

        # Load optimizer
        if load_optimizer and optimizer is not None:
            opt_path = ckpt_dir / "optimizer.pt"
            if opt_path.exists():
                state = torch.load(opt_path, map_location="cpu", weights_only=True)
                optimizer.load_state_dict(state["optimizer"])
                if scheduler and "scheduler" in state:
                    scheduler.load_state_dict(state["scheduler"])
                logger.info("Loaded optimizer state from %s", opt_path)

        return meta

    def _prune(self) -> None:
        """Remove old checkpoints beyond max_checkpoints."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return

        # Sort by step, keep latest
        self.checkpoints.sort(key=lambda c: c.step)
        to_remove = self.checkpoints[: -self.max_checkpoints]
        self.checkpoints = self.checkpoints[-self.max_checkpoints :]

        for ckpt in to_remove:
            if ckpt.path.exists():
                shutil.rmtree(ckpt.path)
                logger.info("Pruned old checkpoint: %s", ckpt.path)

    def get_latest(self) -> Optional[Path]:
        """Get the path of the latest checkpoint."""
        if not self.checkpoints:
            # Scan output directory
            ckpt_dirs = sorted(self.output_dir.glob("checkpoint-*"))
            if ckpt_dirs:
                return ckpt_dirs[-1]
            return None
        return self.checkpoints[-1].path
