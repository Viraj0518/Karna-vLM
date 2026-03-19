"""
Perceiver Resampler bridge.

Inspired by Flamingo's Perceiver Resampler. Uses learned latent queries
with cross-attention to vision features. Simpler than Q-Former (no
self-attention between layers by default), focusing purely on
information extraction from vision tokens.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from karna_vlm.models.bridge.bridge_interface import BridgeInterface, BridgeOutput
from karna_vlm.models.vision.encoder_interface import VisionEncoderOutput


class ResamplerLayer(nn.Module):
    """Single resampler layer: cross-attention + FFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        ffn_dim = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            latents: [B, L, D] learned latent queries
            context: [B, S, D] vision features
            key_padding_mask: Optional [B, S] (True = ignore)
        """
        # Cross-attention: latents attend to context
        residual = latents
        latents = self.norm1(latents)
        latents, _ = self.cross_attn(
            query=latents,
            key=context,
            value=context,
            key_padding_mask=key_padding_mask,
        )
        latents = residual + latents

        # FFN
        residual = latents
        latents = residual + self.ffn(self.norm2(latents))

        return latents


class ResamplerBridge(BridgeInterface):
    """Perceiver Resampler bridge.

    Args:
        vision_dim: Dimension of vision encoder outputs.
        decoder_dim: Dimension of decoder embedding space.
        bridge_dim: Internal dimension (default 512).
        num_queries: Number of latent queries / output tokens (default 64).
        num_layers: Number of resampler layers (default 4).
        num_heads: Number of attention heads (default 8).
        ffn_ratio: FFN expansion ratio (default 4.0).
        dropout: Dropout rate (default 0.1).
    """

    def __init__(
        self,
        vision_dim: int,
        decoder_dim: int,
        bridge_dim: int = 512,
        num_queries: int = 64,
        num_layers: int = 4,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(vision_dim, decoder_dim, num_output_tokens=num_queries)
        self.bridge_dim = bridge_dim

        # Input projection
        self.input_proj = nn.Linear(vision_dim, bridge_dim)
        self.input_norm = nn.LayerNorm(bridge_dim)

        # Learned latent queries
        self.latents = nn.Parameter(torch.randn(1, num_queries, bridge_dim) * 0.02)

        # Temporal/positional encoding for vision features
        self.vision_pos_embed = nn.Parameter(
            torch.randn(1, 1024, bridge_dim) * 0.02  # supports up to 1024 patches
        )

        # Resampler layers
        self.layers = nn.ModuleList([
            ResamplerLayer(bridge_dim, num_heads, ffn_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(bridge_dim)
        self.output_proj = nn.Linear(bridge_dim, decoder_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def bridge(
        self,
        vision_output: VisionEncoderOutput,
        instruction_embeds: Optional[torch.Tensor] = None,
    ) -> BridgeOutput:
        """Resample vision features into fixed-length latent representation.

        Args:
            vision_output: Patch features from vision encoder.
            instruction_embeds: Ignored.

        Returns:
            BridgeOutput with ``num_queries`` output tokens.
        """
        B, S, _ = vision_output.patch_features.shape

        # Project into bridge space
        context = self.input_proj(vision_output.patch_features)  # [B, S, bridge_dim]
        context = self.input_norm(context)

        # Add positional encoding (truncate or pad as needed)
        pos = self.vision_pos_embed[:, :S, :]
        context = context + pos

        # Expand latents
        latents = self.latents.expand(B, -1, -1)

        # Convert attention mask to key_padding_mask format (True = ignore)
        kpm = None
        if vision_output.attention_mask is not None:
            kpm = ~vision_output.attention_mask.bool()

        # Process through resampler layers
        for layer in self.layers:
            latents = layer(latents, context, kpm)

        # Output projection
        latents = self.output_norm(latents)
        output = self.output_proj(latents)

        return BridgeOutput(
            projected_features=output,
            num_tokens=self.num_output_tokens,
        )
