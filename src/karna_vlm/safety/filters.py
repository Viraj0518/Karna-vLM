"""
Content filtering for VLM inputs and outputs.

Provides image-level and text-level content filtering.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of content filtering."""

    passed: bool
    reason: str = ""
    confidence: float = 0.0
    category: str = ""


class ContentFilter:
    """Content filter for images and text.

    Provides basic image validation and text filtering.
    For production, integrate with dedicated content moderation
    APIs (Azure Content Safety, AWS Rekognition, etc.).

    Args:
        max_image_size: Maximum allowed image dimension.
        min_image_size: Minimum allowed image dimension.
        blocked_words: Set of blocked words/phrases.
    """

    def __init__(
        self,
        max_image_size: int = 4096,
        min_image_size: int = 16,
        blocked_words: Optional[set[str]] = None,
    ) -> None:
        self.max_image_size = max_image_size
        self.min_image_size = min_image_size
        self.blocked_words = blocked_words or set()

    def filter_image(self, image: Image.Image) -> FilterResult:
        """Validate an input image.

        Args:
            image: PIL image to validate.

        Returns:
            FilterResult.
        """
        w, h = image.size

        if w > self.max_image_size or h > self.max_image_size:
            return FilterResult(
                passed=False,
                reason=f"Image too large: {w}x{h} (max {self.max_image_size})",
                category="size",
            )

        if w < self.min_image_size or h < self.min_image_size:
            return FilterResult(
                passed=False,
                reason=f"Image too small: {w}x{h} (min {self.min_image_size})",
                category="size",
            )

        if image.mode not in ("RGB", "RGBA", "L"):
            return FilterResult(
                passed=False,
                reason=f"Unsupported image mode: {image.mode}",
                category="format",
            )

        return FilterResult(passed=True)

    def filter_text(self, text: str) -> FilterResult:
        """Filter text content.

        Args:
            text: Text to check.

        Returns:
            FilterResult.
        """
        text_lower = text.lower()

        for word in self.blocked_words:
            if word.lower() in text_lower:
                return FilterResult(
                    passed=False,
                    reason=f"Blocked content detected",
                    category="blocked_word",
                )

        return FilterResult(passed=True)

    def filter_output(self, output: str) -> FilterResult:
        """Filter model output before returning to user.

        Args:
            output: Generated text.

        Returns:
            FilterResult.
        """
        # Basic output validation
        if not output or not output.strip():
            return FilterResult(
                passed=False,
                reason="Empty output",
                category="quality",
            )

        # Check for excessive repetition (hallucination indicator)
        words = output.split()
        if len(words) > 10:
            unique_ratio = len(set(words)) / len(words)
            if unique_ratio < 0.2:
                return FilterResult(
                    passed=False,
                    reason="Excessive repetition detected (possible hallucination)",
                    category="quality",
                    confidence=1 - unique_ratio,
                )

        return FilterResult(passed=True)
