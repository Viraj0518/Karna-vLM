"""
Optional output heads for structured prediction.

These sit on top of the decoder's hidden states to produce
structured outputs like bounding boxes, JSON schemas, etc.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class StructuredOutputHead(nn.Module):
    """Head for structured JSON-like output extraction.

    Maps decoder hidden states to a set of predefined fields
    with confidence scores. Useful for document extraction tasks.

    Args:
        hidden_dim: Decoder hidden dimension.
        num_fields: Maximum number of output fields.
        field_dim: Dimension per field embedding.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_fields: int = 64,
        field_dim: int = 256,
    ) -> None:
        super().__init__()
        self.field_queries = nn.Parameter(torch.randn(num_fields, field_dim) * 0.02)
        self.proj = nn.Linear(hidden_dim, field_dim)
        self.confidence_head = nn.Linear(field_dim, 1)
        self.value_head = nn.Linear(field_dim, hidden_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        field_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: [B, seq_len, hidden_dim] decoder output.
            field_mask: Optional [num_fields] mask.

        Returns:
            Dict with 'field_logits' [B, num_fields, hidden_dim],
            'confidence' [B, num_fields].
        """
        # Project hidden states
        h = self.proj(hidden_states)  # [B, seq_len, field_dim]

        # Cross-attention: fields attend to sequence
        # [num_fields, field_dim] x [B, field_dim, seq_len] -> [B, num_fields, seq_len]
        attn_logits = torch.einsum("fd,bsd->bfs", self.field_queries, h)
        attn_weights = torch.softmax(attn_logits, dim=-1)

        # Aggregate: [B, num_fields, seq_len] x [B, seq_len, hidden_dim]
        field_repr = torch.bmm(attn_weights, hidden_states)  # [B, num_fields, hidden_dim]

        confidence = self.confidence_head(self.proj(field_repr)).squeeze(-1)  # [B, num_fields]
        value_logits = self.value_head(self.proj(field_repr))  # [B, num_fields, hidden_dim]

        return {
            "field_representations": field_repr,
            "confidence": torch.sigmoid(confidence),
            "value_logits": value_logits,
        }


class GroundingHead(nn.Module):
    """Head for visual grounding (bounding box prediction).

    Predicts normalized bounding boxes [x1, y1, x2, y2] from decoder
    hidden states at designated grounding positions.

    Args:
        hidden_dim: Decoder hidden dimension.
        num_bins: Number of coordinate bins for discretized prediction (default 1000).
    """

    def __init__(self, hidden_dim: int, num_bins: int = 1000) -> None:
        super().__init__()
        self.num_bins = num_bins
        self.bbox_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 4 * num_bins),  # 4 coordinates × num_bins
        )

    def forward(self, hidden_states: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            hidden_states: [B, num_boxes, hidden_dim] at grounding positions.

        Returns:
            Dict with 'bbox_logits' [B, num_boxes, 4, num_bins]
            and 'bbox_coords' [B, num_boxes, 4] (normalized 0-1).
        """
        logits = self.bbox_head(hidden_states)  # [B, N, 4*num_bins]
        B, N, _ = logits.shape
        logits = logits.view(B, N, 4, self.num_bins)

        # Convert to coordinates via argmax (inference) or expected value (soft)
        probs = torch.softmax(logits, dim=-1)
        bins = torch.arange(self.num_bins, device=logits.device, dtype=logits.dtype)
        coords = (probs * bins).sum(dim=-1) / self.num_bins  # [B, N, 4]

        return {
            "bbox_logits": logits,
            "bbox_coords": coords,
        }
