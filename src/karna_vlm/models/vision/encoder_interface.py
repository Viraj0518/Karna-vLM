"""
Abstract interface for vision encoders.

All vision backends must implement this interface so they can be
swapped without touching the rest of the pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn
from PIL import Image


@dataclass
class VisionEncoderOutput:
    """Standardized output from any vision encoder.

    Attributes:
        patch_features: Patch-level features [B, num_patches, hidden_dim].
        pooled_features: Optional pooled (CLS) features [B, hidden_dim].
        hidden_dim: Dimensionality of the feature vectors.
        grid_size: Spatial grid (H_patches, W_patches) if applicable.
        attention_mask: Optional mask for valid patches [B, num_patches].
    """

    patch_features: torch.Tensor
    pooled_features: Optional[torch.Tensor] = None
    hidden_dim: int = 0
    grid_size: tuple[int, int] = (0, 0)
    attention_mask: Optional[torch.Tensor] = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.hidden_dim == 0:
            self.hidden_dim = self.patch_features.shape[-1]


class VisionEncoderInterface(ABC, nn.Module):
    """Abstract base class for all vision encoders in Karna VLM.

    Concrete implementations wrap specific pretrained backbones
    (SigLIP, CLIP, EVA-CLIP, etc.) behind this common interface.
    """

    def __init__(self, freeze: bool = True) -> None:
        super().__init__()
        self._frozen = freeze

    # ------------------------------------------------------------------
    # Abstract methods every backend must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def encode(self, pixel_values: torch.Tensor) -> VisionEncoderOutput:
        """Encode images into patch-level features.

        Args:
            pixel_values: Preprocessed images [B, C, H, W].

        Returns:
            VisionEncoderOutput with at minimum ``patch_features`` populated.
        """
        ...

    @abstractmethod
    def get_output_dim(self) -> int:
        """Return the hidden dimension of the encoder's output features."""
        ...

    @abstractmethod
    def get_image_size(self) -> int:
        """Return the expected input image resolution (assumes square)."""
        ...

    @abstractmethod
    def get_num_patches(self) -> int:
        """Return the number of output patches for a single image."""
        ...

    @abstractmethod
    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        """Preprocess PIL images into the tensor format expected by ``encode``.

        Args:
            images: List of PIL images.

        Returns:
            Batched tensor [B, C, H, W].
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def freeze(self) -> None:
        """Freeze all encoder parameters (no gradient computation)."""
        self._frozen = True
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all encoder parameters."""
        self._frozen = False
        for param in self.parameters():
            param.requires_grad = True

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def forward(self, pixel_values: torch.Tensor) -> VisionEncoderOutput:
        """Forward pass (delegates to ``encode``).

        If the encoder is frozen, runs under ``torch.no_grad()``.
        """
        if self._frozen:
            with torch.no_grad():
                return self.encode(pixel_values)
        return self.encode(pixel_values)
