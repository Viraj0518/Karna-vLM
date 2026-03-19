"""
CLIP / EVA-CLIP vision encoder backend.

Wraps HuggingFace ``CLIPVisionModel`` behind the Karna encoder interface.
Supports standard OpenAI CLIP and EVA-CLIP variants.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from PIL import Image

from karna_vlm.models.vision.encoder_interface import VisionEncoderInterface, VisionEncoderOutput

logger = logging.getLogger(__name__)


class CLIPEncoder(VisionEncoderInterface):
    """CLIP-based vision encoder.

    Args:
        model_name: HuggingFace model identifier.
        freeze: Whether to freeze encoder weights (default True).
        select_layer: Hidden layer index to extract features from.
        use_cls: Whether to include CLS token in patch features.
        device: Target device.
    """

    DEFAULT_MODEL = "openai/clip-vit-base-patch16"

    def __init__(
        self,
        model_name: Optional[str] = None,
        freeze: bool = True,
        select_layer: int = -2,
        use_cls: bool = False,
        device: Optional[str] = None,
    ) -> None:
        super().__init__(freeze=freeze)
        self.model_name = model_name or self.DEFAULT_MODEL
        self.select_layer = select_layer
        self.use_cls = use_cls

        from transformers import CLIPImageProcessor, CLIPVisionModel

        logger.info("Loading CLIP encoder: %s", self.model_name)
        self.model: CLIPVisionModel = CLIPVisionModel.from_pretrained(self.model_name)
        self.processor: CLIPImageProcessor = CLIPImageProcessor.from_pretrained(self.model_name)

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
        """Encode images through CLIP vision tower.

        Args:
            pixel_values: [B, C, H, W] preprocessed images.

        Returns:
            VisionEncoderOutput with patch features (CLS excluded by default).
        """
        outputs = self.model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states[self.select_layer]  # [B, 1+num_patches, dim]

        # Separate CLS and patch tokens
        cls_token = hidden_states[:, 0:1, :]  # [B, 1, dim]
        patch_tokens = hidden_states[:, 1:, :]  # [B, num_patches, dim]

        if self.use_cls:
            features = hidden_states  # include CLS
        else:
            features = patch_tokens

        pooled = cls_token.squeeze(1)  # [B, dim]

        grid_side = self._image_size // self._patch_size
        return VisionEncoderOutput(
            patch_features=features,
            pooled_features=pooled,
            hidden_dim=self._hidden_dim,
            grid_size=(grid_side, grid_side),
        )

    def get_output_dim(self) -> int:
        return self._hidden_dim

    def get_image_size(self) -> int:
        return self._image_size

    def get_num_patches(self) -> int:
        return self._num_patches + (1 if self.use_cls else 0)

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        """Preprocess PIL images for CLIP.

        Args:
            images: List of PIL images.

        Returns:
            Tensor [B, C, H, W].
        """
        processed = self.processor(images=images, return_tensors="pt")
        pixel_values: torch.Tensor = processed["pixel_values"]
        device = next(self.parameters()).device
        return pixel_values.to(device=device, dtype=next(self.parameters()).dtype)
