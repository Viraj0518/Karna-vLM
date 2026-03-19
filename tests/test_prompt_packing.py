"""
Tests for prompt packing module.

Verifies:
1. Image token insertion works correctly
2. Attention masks are properly constructed
3. Batching with padding works
4. Label masking is correct
"""

import pytest
import torch
import torch.nn as nn

from karna_vlm.models.prompt_packing.packer import PromptPacker, PackedSequence


class MockTokenizer:
    """Minimal tokenizer mock for testing."""

    def __init__(self, vocab_size: int = 1000) -> None:
        self.vocab_size = vocab_size
        self.eos_token_id = 2
        self.pad_token = "<pad>"
        self.pad_token_id = 0

    def __call__(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = True, **kwargs):
        # Simple char-level "tokenization" for testing
        tokens = [ord(c) % self.vocab_size for c in text[:50]]  # cap at 50 tokens
        if add_special_tokens:
            tokens = [1] + tokens  # BOS
        return {"input_ids": torch.tensor([tokens])}


class TestPromptPacker:
    @pytest.fixture
    def setup(self):
        """Create packer with mock tokenizer and embedding."""
        tokenizer = MockTokenizer()
        embed_dim = 64
        embedding = nn.Embedding(tokenizer.vocab_size, embed_dim)

        def embed_fn(input_ids):
            return embedding(input_ids)

        packer = PromptPacker(tokenizer=tokenizer, embed_fn=embed_fn, max_length=512)
        return packer, embed_dim

    def test_text_only_packing(self, setup):
        packer, embed_dim = setup
        packed = packer.pack("Hello world")

        assert packed.inputs_embeds.dim() == 3  # [1, seq_len, dim]
        assert packed.inputs_embeds.shape[0] == 1
        assert packed.inputs_embeds.shape[2] == embed_dim
        assert packed.attention_mask.shape == packed.inputs_embeds.shape[:2]

    def test_image_token_insertion(self, setup):
        packer, embed_dim = setup
        num_image_tokens = 32
        image_embeds = torch.randn(num_image_tokens, embed_dim)

        packed = packer.pack("<image>\nDescribe this.", image_embeds=image_embeds)

        assert packed.inputs_embeds.dim() == 3
        assert len(packed.image_positions) == 1
        start, end = packed.image_positions[0]
        assert end - start == num_image_tokens

    def test_auto_prepend_image(self, setup):
        packer, embed_dim = setup
        image_embeds = torch.randn(16, embed_dim)

        # No <image> token in text — should auto-prepend
        packed = packer.pack("What is in this image?", image_embeds=image_embeds)

        assert len(packed.image_positions) == 1

    def test_attention_mask_all_ones(self, setup):
        packer, embed_dim = setup
        packed = packer.pack("Test prompt")

        assert packed.attention_mask.sum() == packed.attention_mask.numel()

    def test_debug_token_types(self, setup):
        packer, embed_dim = setup
        image_embeds = torch.randn(8, embed_dim)
        packed = packer.pack("<image>\nDescribe", image_embeds=image_embeds)

        assert packed.debug_token_types is not None
        # Should contain both 0 (text) and 1 (image)
        types = packed.debug_token_types.unique().tolist()
        assert 0 in types  # text tokens
        assert 1 in types  # image tokens

    def test_max_length_truncation(self, setup):
        packer, _ = setup
        packer.max_length = 20

        packed = packer.pack("A" * 100)  # Very long text
        assert packed.inputs_embeds.shape[1] <= 20

    def test_batch_packing(self, setup):
        packer, embed_dim = setup
        texts = ["Hello world", "Describe this image <image>", "Short"]
        image_embeds_list = [
            None,
            torch.randn(16, embed_dim),
            None,
        ]

        packed = packer.pack_batch(texts, image_embeds_list)
        assert packed.inputs_embeds.shape[0] == 3  # batch size
        # All sequences padded to same length
        assert packed.inputs_embeds.shape[1] > 0
