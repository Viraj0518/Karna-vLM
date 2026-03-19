"""
Data collators for batching VLM samples.

Handles image preprocessing, prompt packing, and padding/truncation
for efficient batched training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
from PIL import Image

from karna_vlm.models.prompt_packing.packer import PromptPacker


@dataclass
class VLMCollator:
    """Collator for VLM training batches.

    Takes raw dataset outputs (dicts with 'image', 'prompt', 'response')
    and produces tensors ready for the model.

    Args:
        vision_encoder: The vision encoder (for preprocessing images).
        bridge: The bridge module (for encoding images).
        packer: The prompt packer.
        max_length: Maximum sequence length.
    """

    vision_encoder: Any  # VisionEncoderInterface
    bridge: Any  # BridgeInterface
    packer: PromptPacker
    max_length: int = 2048

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a batch of VLM samples.

        Args:
            batch: List of dicts from VLMDataset.__getitem__.

        Returns:
            Dict with inputs_embeds, attention_mask, labels tensors.
        """
        images = [item["image"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        responses = [item["response"] for item in batch]

        # Preprocess and encode images
        pixel_values = self.vision_encoder.preprocess(images)
        with torch.no_grad():
            vision_out = self.vision_encoder(pixel_values)
        bridge_out = self.bridge(vision_out)

        # Pack each example
        packed_list = []
        for i in range(len(batch)):
            packed = self.packer.pack(
                text=prompts[i],
                image_embeds=bridge_out.projected_features[i],
                labels_text=responses[i],
                mask_image_in_labels=True,
                mask_prompt_in_labels=True,
            )
            packed_list.append(packed)

        # Pad to max length in batch
        max_len = min(
            max(p.inputs_embeds.shape[1] for p in packed_list),
            self.max_length,
        )
        dim = packed_list[0].inputs_embeds.shape[-1]
        device = packed_list[0].inputs_embeds.device
        dtype = packed_list[0].inputs_embeds.dtype
        B = len(packed_list)

        batched_embeds = torch.zeros(B, max_len, dim, device=device, dtype=dtype)
        batched_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)
        batched_labels = torch.full(
            (B, max_len), -100, dtype=torch.long, device=device
        )

        for i, p in enumerate(packed_list):
            length = min(p.inputs_embeds.shape[1], max_len)
            batched_embeds[i, :length] = p.inputs_embeds[0, :length]
            batched_mask[i, :length] = p.attention_mask[0, :length]
            if p.labels is not None:
                batched_labels[i, :length] = p.labels[0, :length]

        return {
            "inputs_embeds": batched_embeds,
            "attention_mask": batched_mask,
            "labels": batched_labels,
        }
