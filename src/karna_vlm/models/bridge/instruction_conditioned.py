"""
Instruction-conditioned bridge.

Extends the Q-Former Lite bridge with instruction awareness: the query
tokens are modulated by text instruction embeddings via cross-attention,
allowing the bridge to extract task-relevant visual information.

This is the most capable bridge variant — best for instruction following
and complex multimodal tasks.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from karna_vlm.models.bridge.bridge_interface import BridgeInterface, BridgeOutput
from karna_vlm.models.bridge.qformer_lite import MultiHeadCrossAttention, QFormerLiteLayer
from karna_vlm.models.vision.encoder_interface import VisionEncoderOutput


class InstructionConditionedLayer(nn.Module):
    """Bridge layer with instruction conditioning.

    1. Self-attention among queries
    2. Cross-attention: queries attend to vision features
    3. Cross-attention: queries attend to instruction embeddings
    4. FFN
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ffn_ratio: float = 4.0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # Core QFormer layer (self-attn + vision cross-attn + FFN)
        self.core_layer = QFormerLiteLayer(dim, num_heads, ffn_ratio, dropout)

        # Instruction cross-attention (additional)
        self.instr_cross_attn = MultiHeadCrossAttention(dim, num_heads, dropout)
        self.instr_norm = nn.LayerNorm(dim)

        # Gating for instruction influence
        self.instr_gate = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        queries: torch.Tensor,
        vision_features: torch.Tensor,
        instruction_embeds: Optional[torch.Tensor] = None,
        vision_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            queries: [B, Q, D]
            vision_features: [B, S, D]
            instruction_embeds: Optional [B, T, D] instruction token embeddings
            vision_mask: Optional [B, S]

        Returns:
            (updated_queries, cross_attn_weights)
        """
        # Core processing (self-attn + vision cross-attn + FFN)
        queries, attn_w = self.core_layer(queries, vision_features, vision_mask)

        # Optional instruction conditioning
        if instruction_embeds is not None:
            residual = queries
            normed = self.instr_norm(queries)
            instr_out, _ = self.instr_cross_attn(normed, instruction_embeds)
            # Gated residual — instruction influence is learnable
            queries = residual + self.instr_gate * instr_out

        return queries, attn_w


class InstructionConditionedBridge(BridgeInterface):
    """Q-Former bridge with instruction-aware conditioning.

    The bridge queries attend to both vision features AND instruction
    embeddings, enabling task-specific visual information extraction.

    Args:
        vision_dim: Vision encoder output dimension.
        decoder_dim: Decoder embedding dimension.
        bridge_dim: Internal bridge dimension (default 512).
        num_queries: Number of output tokens (default 64).
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

        # Input projections
        self.vision_proj = nn.Linear(vision_dim, bridge_dim)
        self.vision_norm = nn.LayerNorm(bridge_dim)

        # Instruction projection (decoder_dim -> bridge_dim, since instructions
        # come from the decoder's tokenizer/embedding layer)
        self.instr_proj = nn.Linear(decoder_dim, bridge_dim)
        self.instr_norm = nn.LayerNorm(bridge_dim)

        # Learnable queries
        self.queries = nn.Parameter(torch.randn(1, num_queries, bridge_dim) * 0.02)
        self.query_pos = nn.Parameter(torch.randn(1, num_queries, bridge_dim) * 0.02)

        # Instruction-conditioned layers
        self.layers = nn.ModuleList([
            InstructionConditionedLayer(bridge_dim, num_heads, ffn_ratio, dropout)
            for _ in range(num_layers)
        ])

        # Output
        self.output_proj = nn.Linear(bridge_dim, decoder_dim)
        self.output_norm = nn.LayerNorm(decoder_dim)

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
        """Process vision features with instruction conditioning.

        Args:
            vision_output: Patch features from vision encoder.
            instruction_embeds: [B, T, decoder_dim] instruction token embeddings.
                If None, behaves like standard Q-Former.

        Returns:
            BridgeOutput with instruction-conditioned visual tokens.
        """
        B = vision_output.patch_features.shape[0]

        # Project vision features
        vision_features = self.vision_proj(vision_output.patch_features)
        vision_features = self.vision_norm(vision_features)

        # Project instruction embeddings if provided
        instr = None
        if instruction_embeds is not None:
            instr = self.instr_proj(instruction_embeds)
            instr = self.instr_norm(instr)

        # Initialize queries
        queries = self.queries.expand(B, -1, -1) + self.query_pos.expand(B, -1, -1)

        # Process through layers
        all_attn = []
        for layer in self.layers:
            queries, attn_w = layer(queries, vision_features, instr, vision_output.attention_mask)
            all_attn.append(attn_w)

        # Output projection
        output = self.output_proj(queries)
        output = self.output_norm(output)

        return BridgeOutput(
            projected_features=output,
            num_tokens=self.num_output_tokens,
            attention_weights=all_attn[-1],
            extra={
                "all_attention_weights": all_attn,
                "instruction_conditioned": instruction_embeds is not None,
            },
        )
