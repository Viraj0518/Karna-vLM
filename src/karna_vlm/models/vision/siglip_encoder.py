"""
SigLIP vision encoder backend.

Wraps HuggingFace ``SiglipVisionModel`` behind the Karna encoder interface.
SigLIP is the default recommended encoder for Karna VLM due to its
strong patch-level representations and efficient architecture.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from PIL import Image

from karna_vlm.models.vision.encoder_interface import VisionEncoderInterface, VisionEncoderOutput

logger = logging.getLogger(__name__)


class SigLIPEncoder(VisionEncoderInterface):
    """SigLIP-based vision encoder.

    Args:
        model_name: HuggingFace model identifier.
        freeze: Whether to freeze encoder weights (default True).
        select_layer: Which hidden layer to extract features from.
            -1 = last layer, -2 = second-to-last, etc.
        device: Target device (unused; kept for API compat).
    """

    # Sensible defaults for common SigLIP variants
    DEFAULT_MODEL = "google/siglip-base-patch16-224"

    def __init__(
        self,
        model_name: Optional[str] = None,
        freeze: bool = True,
        select_layer: int = -1,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(freeze=freeze)
        self.model_name = model_name or self.DEFAULT_MODEL
        self.select_layer = select_layer
        # Note: `device` arg is accepted for API compatibility but not used here.
        # The caller should move the module to the desired device after construction
        # (e.g. model.to(device)).  Preprocessing always returns CPU tensors so
        # that DataLoader workers (which have no GPU access) can call preprocess()
        # safely.

        # Lazy imports to keep the module importable without transformers installed
        from transformers import SiglipImageProcessor, SiglipVisionModel

        logger.info("Loading SigLIP encoder: %s", self.model_name)
        self.model: SiglipVisionModel = SiglipVisionModel.from_pretrained(self.model_name)
        self.processor: SiglipImageProcessor = SiglipImageProcessor.from_pretrained(self.model_name)

        # Cache architecture info
        config = self.model.config
        self._hidden_dim: int = config.hidden_size
        self._image_size: int = config.image_size
        self._patch_size: int = config.patch_size
        self._num_patches: int = (self._image_size // self._patch_size) ** 2

        if freeze:
            self.freeze()

        if device:
            self.to(device)

    def encode(self, pixel_values: torch.Tensor) -> VisionEncoderOutput:
        """Encode images through SigLIP.

        Args:
            pixel_values: [B, C, H, W] preprocessed images.

        Returns:
            VisionEncoderOutput with patch-level and optionally pooled features.
        """
        outputs = self.model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        # Select the desired hidden layer
        hidden_states = outputs.hidden_states[self.select_layer]  # [B, num_patches, dim]

        # SigLIP doesn't have a CLS token — pooled = mean of patches
        pooled = hidden_states.mean(dim=1)  # [B, dim]

        grid_side = self._image_size // self._patch_size
        return VisionEncoderOutput(
            patch_features=hidden_states,
            pooled_features=pooled,
            hidden_dim=self._hidden_dim,
            grid_size=(grid_side, grid_side),
        )

    def get_output_dim(self) -> int:
        return self._hidden_dim

    def get_image_size(self) -> int:
        return self._image_size

    def get_num_patches(self) -> int:
        return self._num_patches

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        """Preprocess PIL images for SigLIP.

        Returns CPU tensors — the training loop is responsible for moving
        pixel values to the correct device.  This keeps preprocessing safe
        inside multi-worker DataLoader workers (which cannot access the GPU).

        Args:
            images: List of PIL images (any size).

        Returns:
            Tensor [B, C, H, W] on CPU with float32 dtype.
        """
        processed = self.processor(images=images, return_tensors="pt")
        # Stay on CPU — caller moves to device as needed
        return processed["pixel_values"]
