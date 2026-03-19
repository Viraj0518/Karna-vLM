"""
Abstract interface for multimodal bridges.

The bridge takes vision encoder output and produces a compact sequence of
embeddings aligned with the decoder's token embedding space.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

from karna_vlm.models.vision.encoder_interface import VisionEncoderOutput


@dataclass
class BridgeOutput:
    """Standardized output from any bridge module.

    Attributes:
        projected_features: Tokens ready for the decoder [B, num_tokens, decoder_dim].
        num_tokens: Number of visual tokens produced per image.
        attention_weights: Optional attention weights for interpretability.
        extra: Additional outputs (gate values, reconstruction losses, etc.).
    """

    projected_features: torch.Tensor
    num_tokens: int = 0
    attention_weights: Optional[torch.Tensor] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_tokens == 0:
            self.num_tokens = self.projected_features.shape[1]


class BridgeInterface(ABC, nn.Module):
    """Abstract base class for all bridge modules.

    Bridges transform vision encoder output into decoder-compatible tokens.
    This is where the proprietary multimodal intelligence lives.
    """

    def __init__(
        self,
        vision_dim: int,
        decoder_dim: int,
        num_output_tokens: int = 64,
    ) -> None:
        super().__init__()
        self.vision_dim = vision_dim
        self.decoder_dim = decoder_dim
        self.num_output_tokens = num_output_tokens

    @abstractmethod
    def bridge(
        self,
        vision_output: VisionEncoderOutput,
        instruction_embeds: Optional[torch.Tensor] = None,
    ) -> BridgeOutput:
        """Transform vision features into decoder-compatible tokens.

        Args:
            vision_output: Output from the vision encoder.
            instruction_embeds: Optional text instruction embeddings
                for instruction-conditioned bridges.

        Returns:
            BridgeOutput with projected features.
        """
        ...

    def forward(
        self,
        vision_output: VisionEncoderOutput,
        instruction_embeds: Optional[torch.Tensor] = None,
    ) -> BridgeOutput:
        """Forward pass (delegates to ``bridge``)."""
        return self.bridge(vision_output, instruction_embeds)

    def get_num_output_tokens(self) -> int:
        """Return the number of visual tokens this bridge produces."""
        return self.num_output_tokens

    def get_trainable_params(self) -> int:
        """Count trainable parameters in this bridge."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
