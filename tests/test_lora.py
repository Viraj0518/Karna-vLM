"""
Tests for LoRA module.

Verifies:
1. LoRALinear produces correct shapes
2. LoRA output is close to original when alpha is small
3. Gradients flow through LoRA path only
"""

import pytest
import torch
import torch.nn as nn

from karna_vlm.training.lora import LoRALinear


class TestLoRALinear:
    def test_output_shape(self) -> None:
        original = nn.Linear(128, 256)
        lora = LoRALinear(original, r=8, alpha=16)

        x = torch.randn(2, 10, 128)
        out = lora(x)
        assert out.shape == (2, 10, 256)

    def test_original_frozen(self) -> None:
        original = nn.Linear(128, 256)
        lora = LoRALinear(original, r=8, alpha=16)

        assert not original.weight.requires_grad
        if original.bias is not None:
            assert not original.bias.requires_grad

    def test_lora_params_trainable(self) -> None:
        original = nn.Linear(128, 256)
        lora = LoRALinear(original, r=8, alpha=16)

        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad

    def test_gradients_flow_through_lora(self) -> None:
        original = nn.Linear(128, 256)
        lora = LoRALinear(original, r=8, alpha=16)

        x = torch.randn(1, 5, 128)
        out = lora(x)
        loss = out.sum()
        loss.backward()

        assert lora.lora_A.grad is not None
        assert lora.lora_B.grad is not None
        # Original should NOT have gradients
        assert original.weight.grad is None

    def test_zero_init_b_means_close_to_original(self) -> None:
        """B is zero-initialized, so initially LoRA output should equal original."""
        original = nn.Linear(64, 128)
        lora = LoRALinear(original, r=4, alpha=8)

        x = torch.randn(1, 3, 64)
        with torch.no_grad():
            original_out = original(x)
            lora_out = lora(x)

        # Should be close since B is zero
        assert torch.allclose(original_out, lora_out, atol=1e-5)

    def test_different_ranks(self) -> None:
        for r in [1, 4, 16, 64]:
            original = nn.Linear(256, 512)
            lora = LoRALinear(original, r=r, alpha=r * 2)

            x = torch.randn(1, 1, 256)
            out = lora(x)
            assert out.shape == (1, 1, 512)

            # Check LoRA A and B shapes
            assert lora.lora_A.shape == (r, 256)
            assert lora.lora_B.shape == (512, r)
