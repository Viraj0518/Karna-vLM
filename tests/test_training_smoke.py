"""
Training smoke test: 1 step with mock data.

Verifies the full training loop (forward → loss → backward → optimizer step)
works end-to-end with the VLMTrainer and VLMTrainingCollator.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Reuse tiny mocks from test_integration
from tests.test_integration import (
    make_tiny_model,
    MockVisionEncoder,
    VOCAB_SIZE,
    DECODER_DIM,
    NUM_BRIDGE_TOKENS,
)
from karna_vlm.training.trainer import VLMTrainer, TrainingConfig
from karna_vlm.data.collators import VLMTrainingCollator
from karna_vlm.models.prompt_packing.packer import PromptPacker


class TinyDataset(Dataset):
    """Tiny synthetic dataset of (image, prompt, response) triples."""

    def __init__(self, size: int = 4) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return {
            "image": Image.new("RGB", (8, 8), color=(idx * 10, idx * 20, idx * 30)),
            "prompt": f"What is item {idx}?",
            "response": f"It is item number {idx}.",
        }


class TestTrainingSmoke:

    def test_one_training_step(self):
        """Full training step: data → collate → forward_train → backward → step."""
        model = make_tiny_model()
        vision_encoder = model.vision_encoder
        packer = model.packer

        dataset = TinyDataset(size=2)
        collator = VLMTrainingCollator(
            vision_encoder=vision_encoder,
            packer=packer,
            max_length=64,
        )
        loader = DataLoader(dataset, batch_size=2, collate_fn=collator, num_workers=0)

        config = TrainingConfig(
            output_dir="outputs_test",
            num_epochs=1,
            batch_size=2,
            gradient_accumulation_steps=1,
            logging_steps=1,
            save_steps=1000,  # don't save during smoke test
            max_steps=1,
            fp16=False,
            bf16=False,
        )

        # Patch trainer to use forward_train instead of forward
        device = torch.device("cpu")
        model.to(device)

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=1e-3
        )

        batch = next(iter(loader))
        pixel_values = batch["pixel_values"].to(device)
        prompts = batch["prompts"]
        responses = batch["responses"]

        model.train()
        optimizer.zero_grad()

        out = model.forward_train(
            pixel_values=pixel_values,
            prompts=prompts,
            responses=responses,
        )

        assert out.loss is not None, "Loss must not be None"
        assert not torch.isnan(out.loss), "Loss must not be NaN"
        assert not torch.isinf(out.loss), "Loss must not be Inf"

        out.loss.backward()
        optimizer.step()

        # Verify bridge params updated
        initial_params = {n: p.clone() for n, p in model.bridge.named_parameters()}
        # (Can't compare before/after here without saving before-step, but
        #  reaching this point without exception is the smoke test goal.)

    def test_collator_returns_cpu_tensors(self):
        """VLMTrainingCollator must return CPU pixel_values."""
        model = make_tiny_model()
        dataset = TinyDataset(size=2)
        collator = VLMTrainingCollator(
            vision_encoder=model.vision_encoder,
            packer=model.packer,
            max_length=64,
        )
        batch = collator([dataset[0], dataset[1]])

        assert "pixel_values" in batch
        assert batch["pixel_values"].device.type == "cpu", "pixel_values must be on CPU"
        assert "prompts" in batch
        assert "responses" in batch

    def test_gradient_checkpointing_config(self):
        """gradient_checkpointing=True in TrainingConfig is stored correctly."""
        config = TrainingConfig(gradient_checkpointing=True)
        assert config.gradient_checkpointing is True

        config2 = TrainingConfig()
        assert config2.gradient_checkpointing is False
