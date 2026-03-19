"""
Prompt packing — the layer that assembles multimodal inputs.

Handles interleaving of text token embeddings and visual token embeddings
from the bridge, creating the final ``inputs_embeds`` tensor that the
decoder consumes. Manages image placeholder tokens, attention masks,
and label masking for training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


# Sentinel token IDs (these are virtual; mapped at embed time)
IMAGE_TOKEN = "<image>"
IMAGE_TOKEN_ID = -200  # Sentinel; replaced during packing


@dataclass
class PackedSequence:
    """Result of packing a multimodal prompt.

    Attributes:
        inputs_embeds: [B, seq_len, hidden_dim] — the fully assembled embeddings.
        attention_mask: [B, seq_len] — 1 for real tokens, 0 for padding.
        labels: Optional [B, seq_len] — training labels (-100 for non-target positions).
        image_positions: List of (start, end) index pairs for image tokens.
        debug_token_types: Optional [B, seq_len] — 0=text, 1=image, 2=padding.
    """

    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    labels: Optional[torch.Tensor] = None
    image_positions: list[tuple[int, int]] = field(default_factory=list)
    debug_token_types: Optional[torch.Tensor] = None


class PromptPacker:
    """Assembles multimodal prompts from text and image components.

    The packer:
    1. Tokenizes text, finding IMAGE_TOKEN placeholders
    2. Embeds text tokens via the decoder's embedding layer
    3. Splices in visual tokens from the bridge at placeholder positions
    4. Builds attention masks and optional training labels
    5. Pads sequences to max_length for batching

    Args:
        tokenizer: HuggingFace tokenizer.
        embed_fn: Callable that maps token IDs to embeddings (decoder.embed_tokens).
        image_token: Special token string for image placeholders.
        max_length: Maximum sequence length.
    """

    def __init__(
        self,
        tokenizer: object,
        embed_fn: callable,
        image_token: str = IMAGE_TOKEN,
        max_length: int = 2048,
    ) -> None:
        self.tokenizer = tokenizer
        self.embed_fn = embed_fn
        self.image_token = image_token
        self.max_length = max_length

    def pack(
        self,
        text: str,
        image_embeds: Optional[torch.Tensor] = None,
        labels_text: Optional[str] = None,
        mask_image_in_labels: bool = True,
        mask_prompt_in_labels: bool = True,
    ) -> PackedSequence:
        """Pack a single text+image example into a sequence.

        Args:
            text: Text prompt, may contain ``<image>`` placeholder.
            image_embeds: [num_image_tokens, hidden_dim] from bridge output.
            labels_text: Optional target text for computing labels.
            mask_image_in_labels: If True, mask image positions in labels with -100.
            mask_prompt_in_labels: If True, mask the prompt portion in labels.

        Returns:
            PackedSequence ready for the decoder.
        """
        device = image_embeds.device if image_embeds is not None else torch.device("cpu")
        dtype = image_embeds.dtype if image_embeds is not None else torch.float32

        # Split text on image placeholder
        parts = text.split(self.image_token)
        num_image_slots = len(parts) - 1

        if image_embeds is not None and num_image_slots == 0:
            # No placeholder found — prepend image tokens
            parts = ["", " " + text]
            num_image_slots = 1

        # Tokenize each text part
        all_embeds: list[torch.Tensor] = []
        all_types: list[int] = []  # 0=text, 1=image
        image_positions: list[tuple[int, int]] = []
        current_pos = 0

        for i, part in enumerate(parts):
            if part:
                tokenizer_out = self.tokenizer(
                    part,
                    return_tensors="pt",
                    add_special_tokens=(i == 0),
                )
                # Support both namespace objects and plain dicts
                if isinstance(tokenizer_out, dict):
                    tokens = tokenizer_out["input_ids"].to(device)
                else:
                    tokens = tokenizer_out.input_ids.to(device)

                text_embeds = self.embed_fn(tokens).squeeze(0)  # [T, dim]
                all_embeds.append(text_embeds.to(dtype))
                all_types.extend([0] * text_embeds.shape[0])
                current_pos += text_embeds.shape[0]

            # Insert image tokens (except after the last text part)
            if i < num_image_slots and image_embeds is not None:
                n_img = image_embeds.shape[0]
                all_embeds.append(image_embeds.to(dtype))
                start = current_pos
                all_types.extend([1] * n_img)
                current_pos += n_img
                image_positions.append((start, current_pos))

        if not all_embeds:
            raise ValueError("Empty prompt after packing. Check input text and image_embeds.")

        # Concatenate
        seq_embeds = torch.cat(all_embeds, dim=0)  # [total_len, dim]
        seq_len = seq_embeds.shape[0]

        # Truncate if necessary
        if seq_len > self.max_length:
            seq_embeds = seq_embeds[: self.max_length]
            all_types = all_types[: self.max_length]
            seq_len = self.max_length

        # Build attention mask (no padding in single-example packing)
        attention_mask = torch.ones(seq_len, dtype=torch.long, device=device)

        # Build labels if requested
        labels = None
        if labels_text is not None:
            full_text = text.replace(self.image_token, "")
            if labels_text not in full_text:
                full_text = full_text + labels_text
            _label_out = self.tokenizer(
                full_text + labels_text,
                return_tensors="pt",
                add_special_tokens=False,
            )
            label_tokens = (_label_out["input_ids"] if isinstance(_label_out, dict) else _label_out.input_ids).squeeze(0)

            # Create label tensor, pad/truncate to seq_len
            labels = torch.full((seq_len,), -100, dtype=torch.long, device=device)
            # Only the response portion gets real labels
            # For simplicity, label the last N tokens
            _resp_out = self.tokenizer(
                labels_text, return_tensors="pt", add_special_tokens=False
            )
            response_tokens = (_resp_out["input_ids"] if isinstance(_resp_out, dict) else _resp_out.input_ids).squeeze(0).to(device)
            n_resp = min(response_tokens.shape[0], seq_len)
            labels[-n_resp:] = response_tokens[:n_resp]

            if mask_image_in_labels:
                for start, end in image_positions:
                    if start < seq_len:
                        labels[start : min(end, seq_len)] = -100

        # Add batch dimension
        debug_types = torch.tensor(all_types, dtype=torch.long, device=device).unsqueeze(0)

        return PackedSequence(
            inputs_embeds=seq_embeds.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            labels=labels.unsqueeze(0) if labels is not None else None,
            image_positions=image_positions,
            debug_token_types=debug_types,
        )

    def pack_batch(
        self,
        texts: list[str],
        image_embeds_list: list[Optional[torch.Tensor]],
        pad_to_max: bool = True,
    ) -> PackedSequence:
        """Pack a batch of examples with padding.

        Args:
            texts: List of text prompts.
            image_embeds_list: List of image embedding tensors (or None per example).
            pad_to_max: Whether to pad to max_length or just max in batch.

        Returns:
            Batched PackedSequence.
        """
        packed = [
            self.pack(text, img_emb)
            for text, img_emb in zip(texts, image_embeds_list)
        ]

        max_len = max(p.inputs_embeds.shape[1] for p in packed)
        if pad_to_max:
            max_len = min(max_len, self.max_length)

        dim = packed[0].inputs_embeds.shape[-1]
        device = packed[0].inputs_embeds.device
        dtype = packed[0].inputs_embeds.dtype
        B = len(packed)

        batched_embeds = torch.zeros(B, max_len, dim, device=device, dtype=dtype)
        batched_mask = torch.zeros(B, max_len, dtype=torch.long, device=device)

        for i, p in enumerate(packed):
            length = min(p.inputs_embeds.shape[1], max_len)
            batched_embeds[i, :length] = p.inputs_embeds[0, :length]
            batched_mask[i, :length] = p.attention_mask[0, :length]

        return PackedSequence(
            inputs_embeds=batched_embeds,
            attention_mask=batched_mask,
        )
