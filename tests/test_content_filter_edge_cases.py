"""
ContentFilter output edge cases — especially repetition detection.
"""

from __future__ import annotations

import pytest

from karna_vlm.safety.filters import ContentFilter


class TestRepetitionDetection:

    def test_highly_repetitive_output_fails(self):
        """100 identical words → unique_ratio < 0.2 → should fail."""
        filt = ContentFilter()
        repetitive = " ".join(["the"] * 100)
        result = filt.filter_output(repetitive)
        assert not result.passed
        assert "repetition" in result.reason.lower()

    def test_near_threshold_repetition(self):
        """Edge case: exactly at the 0.2 unique-ratio boundary."""
        filt = ContentFilter()
        # 5 unique words × 5 repetitions each = 25 words, ratio = 5/25 = 0.2
        # unique_ratio < 0.2 is the condition, so exactly 0.2 should PASS
        words = ["alpha", "beta", "gamma", "delta", "epsilon"] * 5
        text = " ".join(words)
        result = filt.filter_output(text)
        assert result.passed, f"Ratio exactly 0.2 should pass (got reason: {result.reason})"

    def test_below_threshold_fails(self):
        """Just below 0.2 unique-ratio → fails."""
        filt = ContentFilter()
        # 2 unique words × 11 occurrences each = 22 words, ratio = 2/22 ≈ 0.09
        words = ["foo", "bar"] * 11
        text = " ".join(words)
        result = filt.filter_output(text)
        assert not result.passed

    def test_short_output_skips_repetition_check(self):
        """Repetition check only fires for len(words) > 10."""
        filt = ContentFilter()
        # 5 identical words — under the 10-word threshold
        short = " ".join(["hello"] * 5)
        result = filt.filter_output(short)
        assert result.passed

    def test_exactly_eleven_words_triggers_check(self):
        """Repetition check fires at len(words) == 11."""
        filt = ContentFilter()
        # 11 identical words → ratio 1/11 ≈ 0.09 → fail
        text = " ".join(["word"] * 11)
        result = filt.filter_output(text)
        assert not result.passed

    def test_diverse_output_passes(self):
        """Normal, diverse text should pass."""
        filt = ContentFilter()
        text = (
            "The quick brown fox jumps over the lazy dog. "
            "In the beginning was the word, and the word was with God. "
            "To be or not to be, that is the question."
        )
        result = filt.filter_output(text)
        assert result.passed

    def test_empty_output_fails(self):
        """Empty string must fail."""
        filt = ContentFilter()
        assert not filt.filter_output("").passed
        assert not filt.filter_output("   ").passed

    def test_confidence_score_for_repetitive(self):
        """Repetitive output should carry non-zero confidence."""
        filt = ContentFilter()
        text = " ".join(["spam"] * 50)
        result = filt.filter_output(text)
        assert not result.passed
        assert result.confidence > 0.0

    def test_repetition_with_mixed_case(self):
        """Repetition detection is case-sensitive (split on whitespace)."""
        filt = ContentFilter()
        # "The" and "the" are different tokens → ratio = 2/100 = 0.02
        mixed = " ".join(["The", "the"] * 50)
        result = filt.filter_output(mixed)
        assert not result.passed


class TestSafetyActionOrdering:
    """Verify SafetyAction IntEnum ordering is numerically correct."""

    def test_refuse_greater_than_warn(self):
        from karna_vlm.safety.policy import SafetyAction
        assert SafetyAction.REFUSE > SafetyAction.WARN

    def test_warn_greater_than_allow(self):
        from karna_vlm.safety.policy import SafetyAction
        assert SafetyAction.WARN > SafetyAction.ALLOW

    def test_redact_between_warn_and_refuse(self):
        from karna_vlm.safety.policy import SafetyAction
        assert SafetyAction.WARN < SafetyAction.REDACT < SafetyAction.REFUSE

    def test_most_severe_wins(self):
        """When two rules fire, the higher-severity action wins."""
        from karna_vlm.safety.policy import SafetyPolicy, SafetyRule, SafetyAction

        policy = SafetyPolicy(rules=[
            SafetyRule(
                name="warn_rule",
                description="warn",
                patterns=[r"trigger"],
                action=SafetyAction.WARN,
            ),
            SafetyRule(
                name="refuse_rule",
                description="refuse",
                patterns=[r"trigger"],
                action=SafetyAction.REFUSE,
            ),
        ])
        result = policy.check_input("trigger both rules")
        assert result.action == SafetyAction.REFUSE
