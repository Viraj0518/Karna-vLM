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
        # Update image context
        if image is not None:
            self.current_image = image

        # Add user message
        self.history.append(ChatMessage(role="user", content=message, image=image))

        # Build prompt from history
        prompt = self._build_prompt()

        # Generate
        images = [self.current_image] if self.current_image else None
        response = self.model.generate(
            images=images,
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

    def _build_prompt(self) -> str:
        """Build full prompt from conversation history."""
        parts = []
        if self.system_prompt:
            parts.append(f"System: {self.system_prompt}")

        for msg in self.history:
            if msg.role == "user":
                if msg.image is not None:
                    parts.append(f"User: <image>\n{msg.content}")
                else:
                    parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                parts.append(f"Assistant: {msg.content}")

        parts.append("Assistant:")
        return "\n".join(parts)

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
