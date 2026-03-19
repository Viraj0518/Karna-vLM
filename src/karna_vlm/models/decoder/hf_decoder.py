"""
HuggingFace causal LM decoder backend.

Wraps any HuggingFace ``AutoModelForCausalLM`` behind the Karna decoder
interface. Supports LoRA integration via PEFT.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

from karna_vlm.models.decoder.decoder_interface import DecoderInterface, DecoderOutput

logger = logging.getLogger(__name__)


class HFDecoder(DecoderInterface):
    """HuggingFace AutoModelForCausalLM decoder wrapper.

    Args:
        model_name: HuggingFace model identifier (e.g., "Qwen/Qwen2-0.5B").
        freeze: Whether to freeze decoder weights.
        torch_dtype: Model dtype (default bfloat16).
        device_map: Device placement strategy.
        trust_remote_code: Whether to trust remote code.
    """

    # Recommended compact decoders by size tier
    TINY = "Qwen/Qwen2-0.5B"
    SMALL = "Qwen/Qwen2-1.5B"
    MID = "Qwen/Qwen2.5-3B"

    def __init__(
        self,
        model_name: Optional[str] = None,
        freeze: bool = True,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[str] = None,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__(freeze=freeze)
        self.model_name = model_name or self.SMALL

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Loading decoder: %s", self.model_name)

        dtype = torch_dtype or torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Cache config
        self._hidden_dim: int = self.model.config.hidden_size
        self._vocab_size: int = self.model.config.vocab_size

        if freeze:
            self.freeze()

    def decode(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Any] = None,
        use_cache: bool = False,
    ) -> DecoderOutput:
        """Run forward pass through the causal LM.

        Args:
            inputs_embeds: [B, seq_len, hidden_dim] mixed embeddings.
            attention_mask: [B, seq_len].
            labels: [B, seq_len] with -100 for positions to ignore.
            past_key_values: KV cache.
            use_cache: Whether to return KV cache.

        Returns:
            DecoderOutput.
        """
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=True,
            return_dict=True,
        )

        return DecoderOutput(
            logits=outputs.logits,
            loss=outputs.loss,
            hidden_states=outputs.hidden_states[-1] if outputs.hidden_states else None,
            past_key_values=outputs.past_key_values if use_cache else None,
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()

    def get_hidden_dim(self) -> int:
        return self._hidden_dim

    def get_vocab_size(self) -> int:
        return self._vocab_size

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed token IDs into the decoder's embedding space."""
        return self.model.get_input_embeddings()(input_ids)
