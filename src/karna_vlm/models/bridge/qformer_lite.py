"""
Q-Former Lite bridge — the default product bridge.

A lightweight query-based bridge inspired by BLIP-2's Q-Former but
designed for compactness. Uses a small set of learnable query tokens
that cross-attend to vision patch features, producing a fixed-length
compact representation.

This is the core moat module — compact, inspectable, trainable.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from karna_vlm.models.bridge.bridge_interface import BridgeInterface, BridgeOutput
from karna_vlm.models.vision.encoder_interface import VisionEncoderOutput


class MultiHeadCrossAttention(nn.Module):
    """Efficient cross-attention for query-to-vision interaction."""

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: [B, Q, D] learnable queries
            key_value: [B, S, D] vision features
            attention_mask: Optional [B, S] mask for padding

        Returns:
            (output [B, Q, D], attention_weights [B, H, Q, S])
        """
        B, Q, D = query.shape
        S = key_value.shape[1]

        q = self.q_proj(query).view(B, Q, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key_value).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(key_value).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # [B, H, Q, S]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            # attention_mask: [B, S] -> [B, 1, 1, S]
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(~mask.bool(), float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_drop(attn_weights)

        out = torch.matmul(attn_weights, v)  # [B, H, Q, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, Q, D)
        out = self.out_proj(out)

        return out, attn_weights


class QFormerLiteLayer(nn.Module):
    """Single transformer layer for the Q-Former Lite bridge.

    Each layer has:
    1. Self-attention among queries
    2. Cross-attention from queries to vision features
    3. Feed-forward network
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.self_attn_norm = nn.LayerNorm(dim)

        # Cross-attention
        self.cross_attn = MultiHeadCrossAttention(dim, num_heads, dropout)
        self.cross_attn_norm = nn.LayerNorm(dim)

        # FFN
        ffn_dim = int(dim * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(
        self,
        queries: torch.Tensor,
        vision_features: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            queries: [B, Q, D]
            vision_features: [B, S, D]
            vision_mask: Optional [B, S]

        Returns:
            (updated_queries [B, Q, D], cross_attn_weights)
        """
        # Self-attention among queries
        residual = queries
        queries = self.self_attn_norm(queries)
        queries, _ = self.self_attn(queries, queries, queries)
        queries = residual + queries

        # Cross-attention to vision features
        residual = queries
        queries_normed = self.cross_attn_norm(queries)
        cross_out, attn_weights = self.cross_attn(queries_normed, vision_features, vision_mask)
        queries = residual + cross_out

        # FFN
        residual = queries
        queries = residual + self.ffn(self.ffn_norm(queries))

        return queries, attn_weights


class QFormerLiteBridge(BridgeInterface):
    """Lightweight Q-Former bridge.

    Uses a small set of learnable query tokens that cross-attend to
    vision patch features through multiple transformer layers, then
    project into the decoder's embedding space.

    Args:
        vision_dim: Dimension of vision encoder outputs.
        decoder_dim: Dimension of decoder embedding space.
        bridge_dim: Internal dimension of the bridge (default 512).
        num_queries: Number of learnable query tokens (default 64).
        num_layers: Number of transformer layers (default 4).
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

        # Input projection: vision_dim -> bridge_dim
        self.input_proj = nn.Linear(vision_dim, bridge_dim)
        self.input_norm = nn.LayerNorm(bridge_dim)

        # Learnable query tokens
        self.queries = nn.Parameter(torch.randn(1, num_queries, bridge_dim) * 0.02)

        # Positional embedding for queries
        self.query_pos = nn.Parameter(torch.randn(1, num_queries, bridge_dim) * 0.02)

        # Transformer layers
        self.layers = nn.ModuleList([
            QFormerLiteLayer(bridge_dim, num_heads, ffn_ratio, dropout)
            for _ in range(num_layers)
        ])

        # Output projection: bridge_dim -> decoder_dim
        self.output_proj = nn.Linear(bridge_dim, decoder_dim)
        self.output_norm = nn.LayerNorm(decoder_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with careful scaling."""
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
        """Process vision features through learnable queries.

        Args:
            vision_output: Patch features from vision encoder.
            instruction_embeds: Ignored in base Q-Former (see InstructionConditioned variant).

        Returns:
            BridgeOutput with ``num_queries`` output tokens.
        """
        B = vision_output.patch_features.shape[0]

        # Project vision features into bridge space
        vision_features = self.input_proj(vision_output.patch_features)
        vision_features = self.input_norm(vision_features)

        # Expand queries for batch
        queries = self.queries.expand(B, -1, -1) + self.query_pos.expand(B, -1, -1)

        # Process through transformer layers
        all_attn_weights = []
        for layer in self.layers:
            queries, attn_w = layer(queries, vision_features, vision_output.attention_mask)
            all_attn_weights.append(attn_w)

        # Project to decoder space
        output = self.output_proj(queries)
        output = self.output_norm(output)

        return BridgeOutput(
            projected_features=output,
            num_tokens=self.num_output_tokens,
            attention_weights=all_attn_weights[-1],  # last layer's attention
            extra={"all_attention_weights": all_attn_weights},
        )
