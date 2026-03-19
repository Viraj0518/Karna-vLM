"""
Tests for evaluation modules.

Verifies metric computation correctness.
"""

import pytest

from karna_vlm.evaluation.vqa import VQAEvaluator
from karna_vlm.evaluation.grounding import GroundingEvaluator
from karna_vlm.evaluation.instruction import InstructionEvaluator


class TestVQAEvaluator:
    def test_exact_match(self) -> None:
        evaluator = VQAEvaluator()
        preds = ["cat", "dog", "blue"]
        gts = ["cat", "dog", "red"]
        metrics = evaluator.evaluate(preds, gts)
        assert metrics.exact_match == pytest.approx(2 / 3, abs=0.01)

    def test_case_insensitive(self) -> None:
        evaluator = VQAEvaluator()
        preds = ["CAT"]
        gts = ["cat"]
        metrics = evaluator.evaluate(preds, gts)
        assert metrics.exact_match == 1.0

    def test_relaxed_accuracy(self) -> None:
        evaluator = VQAEvaluator()
        preds = ["The color is blue"]
        gts = ["blue"]
        metrics = evaluator.evaluate(preds, gts)
        assert metrics.relaxed_accuracy == 1.0

    def test_empty_input(self) -> None:
        evaluator = VQAEvaluator()
        metrics = evaluator.evaluate([], [])
        assert metrics.num_samples == 0

    def test_multiple_ground_truths(self) -> None:
        evaluator = VQAEvaluator()
        preds = ["2"]
        gts = [["2", "two", "Two"]]
        metrics = evaluator.evaluate(preds, gts)
        assert metrics.exact_match == 1.0


class TestGroundingEvaluator:
    def test_perfect_iou(self) -> None:
        evaluator = GroundingEvaluator()
        pred = [[0.1, 0.1, 0.5, 0.5]]
        gt = [[0.1, 0.1, 0.5, 0.5]]
        metrics = evaluator.evaluate(pred, gt)
        assert metrics.mean_iou == pytest.approx(1.0, abs=0.01)
        assert metrics.precision_at_50 == 1.0

    def test_zero_iou(self) -> None:
        evaluator = GroundingEvaluator()
        pred = [[0.0, 0.0, 0.1, 0.1]]
        gt = [[0.5, 0.5, 0.9, 0.9]]
        metrics = evaluator.evaluate(pred, gt)
        assert metrics.mean_iou == 0.0

    def test_partial_overlap(self) -> None:
        evaluator = GroundingEvaluator()
        pred = [[0.0, 0.0, 0.5, 0.5]]
        gt = [[0.25, 0.25, 0.75, 0.75]]
        metrics = evaluator.evaluate(pred, gt)
        assert 0.0 < metrics.mean_iou < 1.0


class TestInstructionEvaluator:
    def test_json_format_compliance(self) -> None:
        evaluator = InstructionEvaluator()
        preds = ['{"key": "value"}', "not json"]
        instrs = ["Return JSON", "Return JSON"]
        metrics = evaluator.evaluate(preds, instrs, expected_formats=["json", "json"])
        assert metrics.format_compliance == 0.5

    def test_relevance_check(self) -> None:
        evaluator = InstructionEvaluator()
        preds = ["The cat is sitting on a blue mat"]
        instrs = ["Describe the cat on the blue mat"]
        metrics = evaluator.evaluate(preds, instrs)
        assert metrics.relevance > 0.0
