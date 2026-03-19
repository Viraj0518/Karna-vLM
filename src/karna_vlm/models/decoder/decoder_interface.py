"""
Abstract interface for decoder language models.

The decoder consumes a mixed sequence of text tokens and visual tokens
(produced by the bridge) and generates text output.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn


@dataclass
class DecoderOutput:
    """Standardized decoder output.

    Attributes:
        logits: Next-token logits [B, seq_len, vocab_size].
        loss: Optional language modeling loss (if labels provided).
        hidden_states: Optional last hidden states [B, seq_len, hidden_dim].
        past_key_values: Optional KV cache for efficient generation.
    """

    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    past_key_values: Optional[Any] = None


class DecoderInterface(ABC, nn.Module):
    """Abstract base class for decoder LLMs in Karna VLM.

    Implementations wrap HuggingFace causal LMs and expose a clean
    interface for consuming multimodal input embeddings.
    """

    def __init__(self, freeze: bool = True) -> None:
        super().__init__()
        self._frozen = freeze

    @abstractmethod
    def decode(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
    ) -> DecoderOutput:
        """Run decoder forward pass on input embeddings.

        Args:
            inputs_embeds: [B, seq_len, hidden_dim] — mixed text + visual tokens.
            attention_mask: [B, seq_len] attention mask.
            labels: [B, seq_len] token labels for computing loss (-100 = ignore).
            past_key_values: KV cache from previous generation step.
            use_cache: Whether to return updated KV cache.

        Returns:
            DecoderOutput with logits and optional loss/hidden_states.
        """
        ...

    @abstractmethod
    def get_input_embeddings(self) -> nn.Embedding:
        """Return the text token embedding layer."""
        ...

    @abstractmethod
    def get_hidden_dim(self) -> int:
        """Return the decoder's hidden dimension."""
        ...

    @abstractmethod
    def get_vocab_size(self) -> int:
        """Return the decoder's vocabulary size."""
        ...

    @abstractmethod
    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed text token IDs into the decoder's embedding space.

        Args:
            input_ids: [B, seq_len] token IDs.

        Returns:
            [B, seq_len, hidden_dim] token embeddings.
        """
        ...

    def freeze(self) -> None:
        """Freeze all decoder parameters."""
        self._frozen = True
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self) -> None:
        """Unfreeze all decoder parameters."""
        self._frozen = False
        for param in self.parameters():
            param.requires_grad = True

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
    ) -> DecoderOutput:
        """Forward pass (delegates to ``decode``)."""
        return self.decode(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )
