"""
Visualization utilities for debugging and interpretability.

Visualize bridge attention, grounding predictions, and
feature activations.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


def visualize_bridge_attention(
    attention_weights: torch.Tensor,
    image: Image.Image,
    query_idx: int = 0,
    output_path: Optional[str] = None,
) -> Image.Image:
    """Visualize bridge cross-attention as a heatmap overlay.

    Args:
        attention_weights: [num_heads, num_queries, num_patches] or averaged.
        image: Original input image.
        query_idx: Which query's attention to visualize.
        output_path: Optional path to save the visualization.

    Returns:
        PIL image with attention overlay.
    """
    import numpy as np

    # Average across heads if needed
    if attention_weights.dim() == 3:
        attn = attention_weights.mean(dim=0)  # [Q, S]
    elif attention_weights.dim() == 4:
        attn = attention_weights.mean(dim=(0, 1))  # average batch and heads
    else:
        attn = attention_weights

    # Get attention for specific query
    if attn.dim() == 2:
        attn_map = attn[query_idx]  # [S]
    else:
        attn_map = attn

    # Reshape to grid
    S = attn_map.shape[0]
    grid_size = int(S ** 0.5)
    if grid_size * grid_size != S:
        logger.warning("Cannot reshape %d patches to square grid", S)
        return image

    attn_grid = attn_map.view(grid_size, grid_size).cpu().numpy()

    # Normalize to 0-255
    attn_grid = (attn_grid - attn_grid.min()) / (attn_grid.max() - attn_grid.min() + 1e-8)
    attn_grid = (attn_grid * 255).astype(np.uint8)

    # Create heatmap
    heatmap = Image.fromarray(attn_grid, mode="L")
    heatmap = heatmap.resize(image.size, Image.BILINEAR)

    # Apply colormap (red overlay)
    heatmap_rgb = Image.new("RGB", image.size)
    heatmap_np = np.array(heatmap)
    heatmap_rgb_np = np.zeros((*heatmap_np.shape, 3), dtype=np.uint8)
    heatmap_rgb_np[:, :, 0] = heatmap_np  # Red channel
    heatmap_rgb = Image.fromarray(heatmap_rgb_np)

    # Blend with original image
    result = Image.blend(image.convert("RGB"), heatmap_rgb, alpha=0.4)

    if output_path:
        result.save(output_path)
        logger.info("Attention visualization saved: %s", output_path)

    return result


def visualize_grounding(
    image: Image.Image,
    boxes: list[list[float]],
    labels: Optional[list[str]] = None,
    output_path: Optional[str] = None,
) -> Image.Image:
    """Draw bounding boxes on an image.

    Args:
        image: Input image.
        boxes: List of [x1, y1, x2, y2] normalized to 0-1.
        labels: Optional labels for each box.
        output_path: Optional save path.

    Returns:
        Image with drawn boxes.
    """
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    w, h = img.size

    colors = ["red", "blue", "green", "yellow", "purple", "orange", "cyan"]

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box[0] * w, box[1] * h, box[2] * w, box[3] * h
        color = colors[i % len(colors)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        if labels and i < len(labels):
            draw.text((x1, y1 - 12), labels[i], fill=color)

    if output_path:
        img.save(output_path)
        logger.info("Grounding visualization saved: %s", output_path)

    return img
