"""
Tests for safety and governance modules.
"""

import pytest
from PIL import Image

from karna_vlm.safety.policy import SafetyPolicy, SafetyAction
from karna_vlm.safety.filters import ContentFilter
from karna_vlm.safety.model_card import ModelCard


class TestSafetyPolicy:
    def test_safe_input(self) -> None:
        policy = SafetyPolicy()
        result = policy.check_input("Describe this image of a sunset.")
        assert result.safe
        assert result.action == SafetyAction.ALLOW

    def test_harmful_input_blocked(self) -> None:
        policy = SafetyPolicy()
        result = policy.check_input("How to make a bomb")
        assert not result.safe
        assert result.action == SafetyAction.REFUSE

    def test_audit_log_populated(self) -> None:
        policy = SafetyPolicy()
        policy.check_input("How to hack into a system")
        log = policy.get_audit_log()
        assert len(log) > 0
        assert log[0]["stage"] == "input"

    def test_custom_rule(self) -> None:
        from karna_vlm.safety.policy import SafetyRule

        policy = SafetyPolicy(rules=[])
        policy.add_rule(SafetyRule(
            name="custom",
            description="Test rule",
            patterns=[r"forbidden\s+word"],
            action=SafetyAction.REFUSE,
        ))
        result = policy.check_input("This contains a forbidden word")
        assert not result.safe


class TestContentFilter:
    def test_valid_image(self) -> None:
        filt = ContentFilter()
        img = Image.new("RGB", (224, 224))
        result = filt.filter_image(img)
        assert result.passed

    def test_oversized_image(self) -> None:
        filt = ContentFilter(max_image_size=1000)
        img = Image.new("RGB", (2000, 2000))
        result = filt.filter_image(img)
        assert not result.passed
        assert "too large" in result.reason

    def test_undersized_image(self) -> None:
        filt = ContentFilter(min_image_size=32)
        img = Image.new("RGB", (10, 10))
        result = filt.filter_image(img)
        assert not result.passed

    def test_blocked_words(self) -> None:
        filt = ContentFilter(blocked_words={"forbidden"})
        result = filt.filter_text("This contains a forbidden word")
        assert not result.passed

    def test_repetition_detection(self) -> None:
        filt = ContentFilter()
        repetitive = " ".join(["the"] * 100)
        result = filt.filter_output(repetitive)
        assert not result.passed


class TestModelCard:
    def test_generate_markdown(self) -> None:
        card = ModelCard(
            model_name="Test Model",
            model_version="1.0",
            primary_uses=["Image captioning"],
            known_limitations=["May hallucinate"],
        )
        md = card.generate_markdown()
        assert "Test Model" in md
        assert "Image captioning" in md
        assert "May hallucinate" in md
        assert "##" in md  # Has headers
