"""
LoRA (Low-Rank Adaptation) management for Karna VLM.

Provides utilities for applying, composing, and managing LoRA adapters
across the model's components (bridge, decoder, or both).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class LoRAConfig:
    """Configuration for a LoRA adapter.

    Attributes:
        r: Rank of the low-rank matrices.
        alpha: Scaling factor (effective scale = alpha / r).
        dropout: Dropout on LoRA paths.
        target_modules: Module names to apply LoRA to.
        modules_to_save: Modules to fully save (not LoRA'd).
    """

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )
    modules_to_save: list[str] = field(default_factory=list)
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


class LoRAManager:
    """Manages LoRA adapters for the VLM.

    Supports:
    - Applying LoRA to decoder, bridge, or both
    - Multiple named adapters (stacking)
    - Adapter composition for multi-domain deployment
    - Save/load individual adapters
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.active_adapters: dict[str, LoRAConfig] = {}

    def apply_decoder_lora(
        self,
        config: Optional[LoRAConfig] = None,
        adapter_name: str = "default",
    ) -> None:
        """Apply LoRA to the decoder LLM.

        Args:
            config: LoRA configuration.
            adapter_name: Name for this adapter.
        """
        config = config or LoRAConfig()

        try:
            from peft import LoraConfig as PeftLoraConfig, get_peft_model

            peft_config = PeftLoraConfig(
                r=config.r,
                lora_alpha=config.alpha,
                lora_dropout=config.dropout,
                target_modules=config.target_modules,
                bias=config.bias,
                task_type=config.task_type,
            )
            self.model.decoder.model = get_peft_model(
                self.model.decoder.model, peft_config
            )
            self.active_adapters[adapter_name] = config
            logger.info(
                "Applied decoder LoRA '%s': r=%d, alpha=%d, targets=%s",
                adapter_name, config.r, config.alpha, config.target_modules,
            )
        except ImportError:
            raise ImportError("peft required. Install: pip install peft")

    def apply_bridge_lora(
        self,
        config: Optional[LoRAConfig] = None,
        adapter_name: str = "bridge_lora",
    ) -> None:
        """Apply LoRA to bridge linear layers.

        Since bridges are custom modules, we apply LoRA manually
        to their linear projections.

        Args:
            config: LoRA configuration.
            adapter_name: Name for this adapter.
        """
        config = config or LoRAConfig(r=8, alpha=16, target_modules=[])

        bridge = self.model.bridge
        replaced = 0

        for name, module in bridge.named_modules():
            if isinstance(module, nn.Linear) and any(
                t in name for t in ("proj", "q_proj", "k_proj", "v_proj", "out_proj")
            ):
                # Wrap with LoRA
                parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
                parent = bridge if not parent_name else dict(bridge.named_modules())[parent_name]
                lora_layer = LoRALinear(module, r=config.r, alpha=config.alpha, dropout=config.dropout)
                setattr(parent, child_name, lora_layer)
                replaced += 1

        self.active_adapters[adapter_name] = config
        logger.info(
            "Applied bridge LoRA '%s': replaced %d linear layers",
            adapter_name, replaced,
        )

    def save_adapter(self, adapter_name: str, path: str | Path) -> None:
        """Save a specific adapter's weights."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Collect LoRA parameters
        lora_state = {}
        for name, param in self.model.named_parameters():
            if "lora" in name.lower() and param.requires_grad:
                lora_state[name] = param.data.clone()

        torch.save(lora_state, path / f"{adapter_name}.pt")
        logger.info("Saved adapter '%s' (%d params) to %s", adapter_name, len(lora_state), path)

    def load_adapter(self, adapter_name: str, path: str | Path) -> None:
        """Load adapter weights from disk."""
        path = Path(path) / f"{adapter_name}.pt"
        state = torch.load(path, map_location="cpu", weights_only=True)

        model_state = dict(self.model.named_parameters())
        loaded = 0
        for key, value in state.items():
            if key in model_state:
                model_state[key].data.copy_(value)
                loaded += 1

        logger.info("Loaded adapter '%s': %d/%d params", adapter_name, loaded, len(state))

    def get_trainable_summary(self) -> dict[str, int]:
        """Get summary of trainable parameters per component."""
        summary = {"bridge": 0, "decoder": 0, "other": 0}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "bridge" in name:
                summary["bridge"] += param.numel()
            elif "decoder" in name:
                summary["decoder"] += param.numel()
            else:
                summary["other"] += param.numel()
        return summary


class LoRALinear(nn.Module):
    """Manual LoRA wrapper for nn.Linear.

    output = W @ x + (alpha/r) * B @ A @ x

    Args:
        original: The original nn.Linear layer.
        r: LoRA rank.
        alpha: LoRA alpha scaling.
        dropout: Dropout on LoRA path.
    """

    def __init__(
        self,
        original: nn.Linear,
        r: int = 16,
        alpha: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.original = original
        self.r = r
        self.scaling = alpha / r

        # Freeze original
        original.weight.requires_grad = False
        if original.bias is not None:
            original.bias.requires_grad = False

        in_features = original.in_features
        out_features = original.out_features

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(r, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
        return base_out + lora_out
