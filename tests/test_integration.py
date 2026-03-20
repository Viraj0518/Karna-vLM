"""
Integration test: tiny model config, one forward pass.

Uses mocked HuggingFace model objects so no real weights are downloaded.
"""

from __future__ import annotations

from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from PIL import Image

from karna_vlm.models.vlm_model import KarnaVLMConfig, KarnaVLM
from karna_vlm.models.vision.encoder_interface import VisionEncoderInterface, VisionEncoderOutput
from karna_vlm.models.bridge.bridge_interface import BridgeInterface, BridgeOutput
from karna_vlm.models.decoder.decoder_interface import DecoderInterface, DecoderOutput
from karna_vlm.models.prompt_packing.packer import PromptPacker


# ── Tiny mock components ──────────────────────────────────────────────────

VISION_DIM = 16
DECODER_DIM = 32
VOCAB_SIZE = 256
NUM_PATCHES = 4
NUM_BRIDGE_TOKENS = 4


class MockTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    pad_token = "<pad>"

    def __call__(self, text: str, return_tensors="pt", add_special_tokens=True, **kwargs):
        tokens = [ord(c) % VOCAB_SIZE for c in text[:20]]
        if add_special_tokens:
            tokens = [1] + tokens
        return {"input_ids": torch.tensor([tokens])}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        """Convert token IDs back to string (trivial stub)."""
        return "mock output"


class MockVisionEncoder(VisionEncoderInterface):
    """Tiny deterministic vision encoder (no real weights)."""

    def __init__(self) -> None:
        super().__init__(freeze=True)
        self._conv = nn.Linear(3, VISION_DIM)  # dummy param so .parameters() works

    def encode(self, pixel_values: torch.Tensor) -> VisionEncoderOutput:
        B = pixel_values.shape[0]
        feats = torch.zeros(B, NUM_PATCHES, VISION_DIM)
        return VisionEncoderOutput(
            patch_features=feats,
            pooled_features=feats.mean(1),
            hidden_dim=VISION_DIM,
            grid_size=(2, 2),
        )

    def preprocess(self, images: list[Image.Image]) -> torch.Tensor:
        B = len(images)
        return torch.zeros(B, 3, 8, 8)

    def get_output_dim(self) -> int:
        return VISION_DIM

    def get_image_size(self) -> int:
        return 8

    def get_num_patches(self) -> int:
        return NUM_PATCHES


class MockBridge(BridgeInterface):
    """Tiny trainable bridge."""

    def __init__(self) -> None:
        super().__init__(vision_dim=VISION_DIM, decoder_dim=DECODER_DIM, num_output_tokens=NUM_BRIDGE_TOKENS)
        self.proj = nn.Linear(VISION_DIM, DECODER_DIM)

    def bridge(self, vision_output: VisionEncoderOutput, instruction_embeds=None) -> BridgeOutput:
        # Use mean-pooled features and expand to NUM_BRIDGE_TOKENS
        B = vision_output.patch_features.shape[0]
        pooled = vision_output.patch_features.mean(1, keepdim=True)  # [B, 1, V]
        projected = self.proj(pooled.expand(-1, NUM_BRIDGE_TOKENS, -1))  # [B, T, D]
        return BridgeOutput(projected_features=projected)


class MockDecoder(DecoderInterface):
    """Tiny causal LM decoder mock."""

    def __init__(self) -> None:
        super().__init__(freeze=False)
        self.tokenizer = MockTokenizer()
        self._embedding = nn.Embedding(VOCAB_SIZE, DECODER_DIM)
        self._lm_head = nn.Linear(DECODER_DIM, VOCAB_SIZE, bias=False)
        self._hidden_dim = DECODER_DIM

    def decode(self, inputs_embeds, attention_mask=None, labels=None,
               past_key_values=None, use_cache=False) -> DecoderOutput:
        # Trivial: project each embed to logits
        logits = self._lm_head(inputs_embeds)  # [B, T, vocab]
        loss = None
        if labels is not None:
            # Shift for causal LM
            shift_logits = logits[:, :-1].contiguous().view(-1, VOCAB_SIZE)
            shift_labels = labels[:, 1:].contiguous().view(-1)
            valid = shift_labels != -100
            if valid.any():
                loss = nn.CrossEntropyLoss()(shift_logits[valid], shift_labels[valid])
            else:
                loss = logits.sum() * 0.0
        return DecoderOutput(logits=logits, loss=loss)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self._embedding(input_ids)

    def get_input_embeddings(self) -> nn.Embedding:
        return self._embedding

    def get_hidden_dim(self) -> int:
        return self._hidden_dim

    def get_vocab_size(self) -> int:
        return VOCAB_SIZE


def make_tiny_model() -> KarnaVLM:
    """Build a KarnaVLM with fully mocked components (no HF downloads)."""
    config = KarnaVLMConfig(
        model_name="karna-vlm-test",
        model_family="tiny",
        bridge_type="linear",
        bridge_num_queries=NUM_BRIDGE_TOKENS,
        max_length=64,
    )
    model = object.__new__(KarnaVLM)
    nn.Module.__init__(model)
    model.config = config

    vision_encoder = MockVisionEncoder()
    bridge = MockBridge()
    decoder = MockDecoder()

    model.vision_encoder = vision_encoder
    model.bridge = bridge
    model.decoder = decoder
    model.packer = PromptPacker(
        tokenizer=decoder.tokenizer,
        embed_fn=decoder.embed_tokens,
        max_length=64,
    )

    # Register sub-modules properly
    model.add_module("vision_encoder", vision_encoder)
    model.add_module("bridge", bridge)
    model.add_module("decoder", decoder)

    return model


# ── Tests ────────────────────────────────────────────────────────────────


class TestIntegration:

    def test_forward_images_and_text(self):
        """Full pipeline: images + text → decoder output."""
        model = make_tiny_model()
        model.eval()

        img = Image.new("RGB", (8, 8), color=(128, 64, 32))
        with torch.no_grad():
            out = model.forward(images=[img], text="Describe this.")

        assert out.logits is not None
        assert out.logits.dim() == 3  # [B, seq_len, vocab]

    def test_forward_inputs_embeds(self):
        """Pre-packed inputs_embeds path."""
        model = make_tiny_model()
        model.eval()

        B, T, D = 2, 10, DECODER_DIM
        embeds = torch.randn(B, T, D)
        mask = torch.ones(B, T, dtype=torch.long)

        with torch.no_grad():
            out = model.forward(inputs_embeds=embeds, attention_mask=mask)

        assert out.logits.shape == (B, T, VOCAB_SIZE)

    def test_forward_train_gradient_flow(self):
        """forward_train() must produce a loss with gradients through bridge."""
        model = make_tiny_model()
        model.train()

        img = Image.new("RGB", (8, 8))
        pixel_values = model.vision_encoder.preprocess([img])

        out = model.forward_train(
            pixel_values=pixel_values,
            prompts=["Describe this."],
            responses=["A red square."],
        )

        assert out.loss is not None
        # Gradients should flow back to bridge
        out.loss.backward()
        bridge_grads = [p.grad for p in model.bridge.parameters() if p.requires_grad]
        assert any(g is not None for g in bridge_grads), "Bridge should receive gradients"

    def test_generate(self):
        """generate() returns a non-empty string."""
        model = make_tiny_model()
        model.eval()

        img = Image.new("RGB", (8, 8))
        result = model.generate(images=[img], prompt="What is this?", max_new_tokens=5)
        assert isinstance(result, str)

    def test_save_load_roundtrip(self, tmp_path):
        """save_pretrained / from_pretrained round-trip preserves bridge weights."""
        model = make_tiny_model()

        # Inject deterministic bridge weights
        for p in model.bridge.parameters():
            nn.init.constant_(p, 0.42)

        model.save_pretrained(tmp_path)
        assert (tmp_path / "bridge_weights.pt").exists()
        assert (tmp_path / "config.yaml").exists()

        # Build a fresh model and load weights
        model2 = make_tiny_model()
        import torch
        state = torch.load(tmp_path / "bridge_weights.pt", map_location="cpu", weights_only=True)
        model2.bridge.load_state_dict(state)

        for p1, p2 in zip(model.bridge.parameters(), model2.bridge.parameters()):
            assert torch.allclose(p1, p2)
