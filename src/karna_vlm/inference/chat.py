"""
Multi-turn chat interface for Karna VLM.

Manages conversation history and image context across turns.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """A single message in a conversation."""

    role: str  # "user" or "assistant"
    content: str
    image: Optional[Image.Image] = None


class ChatSession:
    """Multi-turn chat session with image context.

    Maintains conversation history and manages image context
    across multiple turns.

    Each user turn may include an image.  The prompt is built with an
    ``<image>`` token for every turn that had an image, and the
    corresponding ordered list of images is passed to ``generate()``
    so that tokens and images stay aligned.

    Args:
        model: KarnaVLM instance.
        system_prompt: Optional system prompt.
        max_history_turns: Maximum number of turns to retain.
        max_new_tokens: Default generation length.
    """

    def __init__(
        self,
        model: Any,
        system_prompt: str = "You are a helpful multimodal assistant.",
        max_history_turns: int = 10,
        max_new_tokens: int = 256,
    ) -> None:
        self.model = model
        self.system_prompt = system_prompt
        self.max_history_turns = max_history_turns
        self.max_new_tokens = max_new_tokens
        self.history: list[ChatMessage] = []
        # Kept for backward compat (single latest image)
        self.current_image: Optional[Image.Image] = None

    def chat(
        self,
        message: str,
        image: Optional[Image.Image] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.7,
    ) -> str:
        """Send a message and get a response.

        Args:
            message: User message text.
            image: Optional new image to discuss.
            max_new_tokens: Override default generation length.
            temperature: Sampling temperature.

        Returns:
            Assistant response text.
        """
        # Update latest-image tracker
        if image is not None:
            self.current_image = image

        # Add user message
        self.history.append(ChatMessage(role="user", content=message, image=image))

        # Build prompt and collect all images (one per turn that has one)
        prompt, images = self._build_prompt_and_images()

        # Generate
        response = self.model.generate(
            images=images if images else None,
            prompt=prompt,
            max_new_tokens=max_new_tokens or self.max_new_tokens,
            temperature=temperature,
        )

        # Add assistant response
        self.history.append(ChatMessage(role="assistant", content=response))

        # Trim history
        if len(self.history) > self.max_history_turns * 2:
            self.history = self.history[-(self.max_history_turns * 2):]

        return response

    def _build_prompt_and_images(self) -> tuple[str, list[Image.Image]]:
        """Build full prompt and ordered image list from conversation history.

        Returns:
            (prompt_str, images_list) — one ``<image>`` token per image in
            the prompt, matched positionally to the returned images list.
        """
        parts = []
        images: list[Image.Image] = []

        if self.system_prompt:
            parts.append(f"System: {self.system_prompt}")

        for msg in self.history:
            if msg.role == "user":
                if msg.image is not None:
                    # Add image token and track the image
                    parts.append(f"User: <image>\n{msg.content}")
                    images.append(msg.image)
                else:
                    parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")

        parts.append("Assistant:")
        return "\n".join(parts), images

    def _build_prompt(self) -> str:
        """Build full prompt from conversation history (legacy, single-image).

        Kept for backward compatibility.  Prefer ``_build_prompt_and_images()``.
        """
        prompt, _ = self._build_prompt_and_images()
        return prompt

    def reset(self) -> None:
        """Clear conversation history."""
        self.history.clear()
        self.current_image = None

    def get_history(self) -> list[dict[str, str]]:
        """Get conversation history as list of dicts."""
        return [
            {"role": msg.role, "content": msg.content}
            for msg in self.history
        ]
