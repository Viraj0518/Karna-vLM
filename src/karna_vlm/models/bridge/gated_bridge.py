"""
Gated bridge — linear projector with learnable per-feature gates.

Adds a sigmoid gating mechanism that learns which visual features
to emphasize or suppress before projecting into decoder space.
More expressive than pure linear but lighter than Q-Former.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from karna_vlm.models.bridge.bridge_interface import BridgeInterface, BridgeOutput
from karna_vlm.models.vision.encoder_interface import VisionEncoderOutput


class GatedBridge(BridgeInterface):
    """Gated linear projection bridge.

    Each patch feature is independently gated and projected:
        gate = sigmoid(W_gate @ x + b_gate)
        output = LayerNorm(MLP(gate * x))

    Args:
        vision_dim: Vision encoder output dimension.
        decoder_dim: Decoder embedding dimension.
        hidden_dim: MLP hidden dimension (default 4× vision_dim).
        dropout: Dropout rate.
    """

    def __init__(
        self,
        vision_dim: int,
        decoder_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        num_output_tokens: int = 0,
    ) -> None:
        super().__init__(vision_dim, decoder_dim, num_output_tokens=num_output_tokens)

        hidden = hidden_dim or vision_dim * 4

        # Gating network
        self.gate_proj = nn.Sequential(
            nn.Linear(vision_dim, vision_dim),
            nn.Sigmoid(),
        )

        # Projection MLP
        self.proj = nn.Sequential(
            nn.Linear(vision_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, decoder_dim),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(decoder_dim)

    def bridge(
        self,
        vision_output: VisionEncoderOutput,
        instruction_embeds: Optional[torch.Tensor] = None,
    ) -> BridgeOutput:
        """Apply gated projection to vision features.

        Args:
            vision_output: Patch features [B, S, vision_dim].
            instruction_embeds: Ignored.

        Returns:
            BridgeOutput with gated projected features.
        """
        features = vision_output.patch_features
        gate_values = self.gate_proj(features)  # [B, S, vision_dim]
        gated = features * gate_values  # element-wise gating
        projected = self.proj(gated)
        projected = self.layer_norm(projected)

        return BridgeOutput(
            projected_features=projected,
            num_tokens=projected.shape[1],
            extra={"gate_values": gate_values},
        )
