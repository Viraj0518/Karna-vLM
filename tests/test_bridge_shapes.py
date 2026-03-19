"""
Shape tests for bridge modules.

Verifies that every bridge variant:
1. Accepts vision encoder output
2. Produces the correct output shape
3. Has the expected number of output tokens
"""

import pytest
import torch

from karna_vlm.models.vision.encoder_interface import VisionEncoderOutput
from karna_vlm.models.bridge.linear_projector import LinearProjector
from karna_vlm.models.bridge.qformer_lite import QFormerLiteBridge
from karna_vlm.models.bridge.resampler import ResamplerBridge
from karna_vlm.models.bridge.gated_bridge import GatedBridge
from karna_vlm.models.bridge.instruction_conditioned import InstructionConditionedBridge


# Common test parameters
BATCH_SIZE = 2
NUM_PATCHES = 196  # 14x14 for a 224px image with 16px patches
VISION_DIM = 768
DECODER_DIM = 1536
BRIDGE_DIM = 512
NUM_QUERIES = 64


def make_vision_output(
    batch_size: int = BATCH_SIZE,
    num_patches: int = NUM_PATCHES,
    dim: int = VISION_DIM,
) -> VisionEncoderOutput:
    """Create a mock VisionEncoderOutput for testing."""
    return VisionEncoderOutput(
        patch_features=torch.randn(batch_size, num_patches, dim),
        pooled_features=torch.randn(batch_size, dim),
        hidden_dim=dim,
        grid_size=(14, 14),
    )


class TestLinearProjector:
    def test_output_shape(self) -> None:
        bridge = LinearProjector(vision_dim=VISION_DIM, decoder_dim=DECODER_DIM)
        vision_out = make_vision_output()
        result = bridge(vision_out)

        assert result.projected_features.shape == (BATCH_SIZE, NUM_PATCHES, DECODER_DIM)
        assert result.num_tokens == NUM_PATCHES

    def test_trainable_params(self) -> None:
        bridge = LinearProjector(vision_dim=VISION_DIM, decoder_dim=DECODER_DIM)
        assert bridge.get_trainable_params() > 0


class TestQFormerLiteBridge:
    def test_output_shape(self) -> None:
        bridge = QFormerLiteBridge(
            vision_dim=VISION_DIM,
            decoder_dim=DECODER_DIM,
            bridge_dim=BRIDGE_DIM,
            num_queries=NUM_QUERIES,
            num_layers=2,
            num_heads=8,
        )
        vision_out = make_vision_output()
        result = bridge(vision_out)

        assert result.projected_features.shape == (BATCH_SIZE, NUM_QUERIES, DECODER_DIM)
        assert result.num_tokens == NUM_QUERIES

    def test_attention_weights_returned(self) -> None:
        bridge = QFormerLiteBridge(
            vision_dim=VISION_DIM,
            decoder_dim=DECODER_DIM,
            bridge_dim=BRIDGE_DIM,
            num_queries=NUM_QUERIES,
            num_layers=2,
        )
        vision_out = make_vision_output()
        result = bridge(vision_out)

        assert result.attention_weights is not None
        assert "all_attention_weights" in result.extra


class TestResamplerBridge:
    def test_output_shape(self) -> None:
        bridge = ResamplerBridge(
            vision_dim=VISION_DIM,
            decoder_dim=DECODER_DIM,
            bridge_dim=BRIDGE_DIM,
            num_queries=NUM_QUERIES,
            num_layers=2,
        )
        vision_out = make_vision_output()
        result = bridge(vision_out)

        assert result.projected_features.shape == (BATCH_SIZE, NUM_QUERIES, DECODER_DIM)
        assert result.num_tokens == NUM_QUERIES

    def test_with_attention_mask(self) -> None:
        bridge = ResamplerBridge(
            vision_dim=VISION_DIM,
            decoder_dim=DECODER_DIM,
            bridge_dim=BRIDGE_DIM,
            num_queries=32,
            num_layers=2,
        )
        vision_out = make_vision_output()
        vision_out.attention_mask = torch.ones(BATCH_SIZE, NUM_PATCHES)
        vision_out.attention_mask[:, -10:] = 0  # mask last 10 patches

        result = bridge(vision_out)
        assert result.projected_features.shape == (BATCH_SIZE, 32, DECODER_DIM)


class TestGatedBridge:
    def test_output_shape(self) -> None:
        bridge = GatedBridge(vision_dim=VISION_DIM, decoder_dim=DECODER_DIM)
        vision_out = make_vision_output()
        result = bridge(vision_out)

        assert result.projected_features.shape == (BATCH_SIZE, NUM_PATCHES, DECODER_DIM)
        assert "gate_values" in result.extra

    def test_gate_values_range(self) -> None:
        bridge = GatedBridge(vision_dim=VISION_DIM, decoder_dim=DECODER_DIM)
        vision_out = make_vision_output()
        result = bridge(vision_out)

        gates = result.extra["gate_values"]
        assert gates.min() >= 0.0
        assert gates.max() <= 1.0


class TestInstructionConditionedBridge:
    def test_output_shape_without_instruction(self) -> None:
        bridge = InstructionConditionedBridge(
            vision_dim=VISION_DIM,
            decoder_dim=DECODER_DIM,
            bridge_dim=BRIDGE_DIM,
            num_queries=NUM_QUERIES,
            num_layers=2,
        )
        vision_out = make_vision_output()
        result = bridge(vision_out)

        assert result.projected_features.shape == (BATCH_SIZE, NUM_QUERIES, DECODER_DIM)

    def test_output_shape_with_instruction(self) -> None:
        bridge = InstructionConditionedBridge(
            vision_dim=VISION_DIM,
            decoder_dim=DECODER_DIM,
            bridge_dim=BRIDGE_DIM,
            num_queries=NUM_QUERIES,
            num_layers=2,
        )
        vision_out = make_vision_output()
        instruction_embeds = torch.randn(BATCH_SIZE, 20, DECODER_DIM)
        result = bridge(vision_out, instruction_embeds=instruction_embeds)

        assert result.projected_features.shape == (BATCH_SIZE, NUM_QUERIES, DECODER_DIM)
        assert result.extra.get("instruction_conditioned") is True

    def test_without_instruction_flag(self) -> None:
        bridge = InstructionConditionedBridge(
            vision_dim=VISION_DIM,
            decoder_dim=DECODER_DIM,
            bridge_dim=BRIDGE_DIM,
            num_queries=NUM_QUERIES,
            num_layers=2,
        )
        vision_out = make_vision_output()
        result = bridge(vision_out)  # No instruction
        assert result.extra.get("instruction_conditioned") is False


class TestAllBridgesGradients:
    """Verify that all bridges produce gradients when trained."""

    @pytest.mark.parametrize("bridge_cls,kwargs", [
        (LinearProjector, {"vision_dim": VISION_DIM, "decoder_dim": DECODER_DIM}),
        (QFormerLiteBridge, {"vision_dim": VISION_DIM, "decoder_dim": DECODER_DIM, "bridge_dim": 256, "num_queries": 16, "num_layers": 1}),
        (ResamplerBridge, {"vision_dim": VISION_DIM, "decoder_dim": DECODER_DIM, "bridge_dim": 256, "num_queries": 16, "num_layers": 1}),
        (GatedBridge, {"vision_dim": VISION_DIM, "decoder_dim": DECODER_DIM}),
        (InstructionConditionedBridge, {"vision_dim": VISION_DIM, "decoder_dim": DECODER_DIM, "bridge_dim": 256, "num_queries": 16, "num_layers": 1}),
    ])
    def test_gradients_flow(self, bridge_cls: type, kwargs: dict) -> None:
        bridge = bridge_cls(**kwargs)
        vision_out = make_vision_output(batch_size=1, num_patches=49, dim=VISION_DIM)

        result = bridge(vision_out)
        loss = result.projected_features.sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in bridge.parameters()
            if p.requires_grad
        )
        assert has_grad, f"No gradients in {bridge_cls.__name__}"
