"""
Linear projector bridge — the simplest baseline.

Projects each vision patch token independently into the decoder's embedding
space via a learnable linear transformation. No cross-attention, no queries.
Fast but limited in its ability to compress or abstract visual information.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from karna_vlm.models.bridge.bridge_interface import BridgeInterface, BridgeOutput
from karna_vlm.models.vision.encoder_interface import VisionEncoderOutput


class LinearProjector(BridgeInterface):
    """Two-layer MLP projector (LLaVA-style).

    Maps each patch feature independently: vision_dim -> hidden -> decoder_dim.
    Output tokens = number of input patches (no compression).

    Args:
        vision_dim: Dimension of vision encoder outputs.
        decoder_dim: Dimension of decoder embedding space.
        hidden_dim: Optional intermediate MLP dimension (default 4× vision_dim).
        dropout: Dropout rate between layers.
    """

    def __init__(
        self,
        vision_dim: int,
        decoder_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        num_output_tokens: int = 0,  # passthrough; determined by encoder
    ) -> None:
        # For linear projector, num_output_tokens is determined by input
        super().__init__(vision_dim, decoder_dim, num_output_tokens=num_output_tokens)

        hidden = hidden_dim or vision_dim * 4
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
        """Project patch features through the MLP.

        Args:
            vision_output: Patch features from the vision encoder.
            instruction_embeds: Ignored (linear projector doesn't use instructions).

        Returns:
            BridgeOutput with one output token per input patch.
        """
        # [B, num_patches, vision_dim] -> [B, num_patches, decoder_dim]
        projected = self.proj(vision_output.patch_features)
        projected = self.layer_norm(projected)

        return BridgeOutput(
            projected_features=projected,
            num_tokens=projected.shape[1],
        )
