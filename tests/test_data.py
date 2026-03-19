"""
Tests for data schemas and templates.
"""

import pytest

from karna_vlm.data.schemas import VLMSample, DatasetManifest, TaskType, License
from karna_vlm.data.templates import PromptTemplate, TEMPLATES, get_template
from karna_vlm.data.mixtures import DatasetMixture, MixtureComponent


class TestVLMSample:
    def test_basic_creation(self) -> None:
        sample = VLMSample(
            image_path="test.jpg",
            conversations=[
                {"role": "user", "content": "What is this?"},
                {"role": "assistant", "content": "A cat."},
            ],
        )
        assert sample.get_prompt() == "What is this?"
        assert sample.get_response() == "A cat."

    def test_to_dict(self) -> None:
        sample = VLMSample(
            image_path="test.jpg",
            conversations=[{"role": "user", "content": "Q"}],
            task_type=TaskType.VQA,
        )
        d = sample.to_dict()
        assert d["image_path"] == "test.jpg"
        assert d["task_type"] == "vqa"


class TestDatasetManifest:
    def test_commercial_safety(self) -> None:
        safe = DatasetManifest(name="safe", license=License.CC_BY_4)
        unsafe = DatasetManifest(name="unsafe", license=License.RESEARCH_ONLY)
        unknown = DatasetManifest(name="unknown", license=License.UNKNOWN)

        assert safe.is_commercial_safe()
        assert not unsafe.is_commercial_safe()
        assert not unknown.is_commercial_safe()


class TestPromptTemplates:
    def test_default_template(self) -> None:
        template = TEMPLATES["default"]
        result = template.format_prompt("Describe this image")
        assert "<image>" in result
        assert "Describe this image" in result

    def test_vqa_template(self) -> None:
        template = TEMPLATES["vqa"]
        result = template.format_prompt("What color is the cat?")
        assert "Question:" in result
        assert "What color is the cat?" in result

    def test_all_templates_exist(self) -> None:
        expected = ["default", "caption", "vqa", "instruction", "chat", "ocr", "grounding"]
        for name in expected:
            assert name in TEMPLATES, f"Missing template: {name}"

    def test_get_template_fallback(self) -> None:
        template = get_template("nonexistent")
        assert template.name == "default"


class TestDatasetMixture:
    def test_mixture_length(self) -> None:
        from torch.utils.data import TensorDataset
        import torch

        ds1 = TensorDataset(torch.randn(10, 5))
        ds2 = TensorDataset(torch.randn(20, 5))

        mixture = DatasetMixture(
            components=[
                MixtureComponent(dataset=ds1, weight=1.0, name="ds1"),
                MixtureComponent(dataset=ds2, weight=2.0, name="ds2"),
            ],
            total_samples=30,
        )
        assert len(mixture) == 30

    def test_mixture_getitem(self) -> None:
        from torch.utils.data import TensorDataset
        import torch

        ds = TensorDataset(torch.randn(5, 3))
        mixture = DatasetMixture(
            components=[MixtureComponent(dataset=ds, weight=1.0, name="test")],
            total_samples=10,
        )
        item = mixture[0]
        assert "mixture_source" in item
        assert item["mixture_source"] == "test"
