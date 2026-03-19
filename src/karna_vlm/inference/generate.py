"""
Generation utilities for Karna VLM.

Provides high-level generation functions with various decoding
strategies and output formats.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


@torch.no_grad()
def generate(
    model: Any,
    image: Optional[Image.Image] = None,
    prompt: str = "Describe this image.",
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    repetition_penalty: float = 1.1,
    **kwargs: Any,
) -> str:
    """Generate text from the VLM.

    Args:
        model: KarnaVLM instance.
        image: Optional PIL image.
        prompt: Text prompt.
        max_new_tokens: Max tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        top_k: Top-k filtering.
        do_sample: Whether to sample.
        repetition_penalty: Penalty for repeated tokens.

    Returns:
        Generated text.
    """
    images = [image] if image is not None else None
    return model.generate(
        images=images,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        **kwargs,
    )


@torch.no_grad()
def batch_generate(
    model: Any,
    images: list[Optional[Image.Image]],
    prompts: list[str],
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    **kwargs: Any,
) -> list[str]:
    """Generate text for a batch of images + prompts.

    Args:
        model: KarnaVLM instance.
        images: List of PIL images (or None for text-only).
        prompts: List of text prompts.
        max_new_tokens: Max tokens per generation.
        temperature: Sampling temperature.

    Returns:
        List of generated texts.
    """
    results = []
    for image, prompt in zip(images, prompts):
        result = generate(
            model=model,
            image=image,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )
        results.append(result)
    return results
