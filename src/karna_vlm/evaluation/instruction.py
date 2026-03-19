"""
Instruction-following evaluation.

Evaluates how well the model follows multimodal instructions
using LLM-as-judge and rule-based metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class InstructionMetrics:
    """Instruction-following evaluation metrics."""

    format_compliance: float = 0.0  # Does output match requested format?
    relevance: float = 0.0  # Is output relevant to the instruction?
    completeness: float = 0.0  # Does output address all parts of instruction?
    overall_score: float = 0.0
    num_samples: int = 0


class InstructionEvaluator:
    """Evaluator for instruction-following quality.

    Uses rule-based heuristics for format compliance and
    optional LLM-as-judge for quality assessment.
    """

    def __init__(self, judge_model: Optional[str] = None) -> None:
        self.judge_model = judge_model

    def evaluate(
        self,
        predictions: list[str],
        instructions: list[str],
        expected_formats: Optional[list[str]] = None,
    ) -> InstructionMetrics:
        """Evaluate instruction-following quality.

        Args:
            predictions: Model outputs.
            instructions: Original instructions.
            expected_formats: Optional expected output formats (e.g., "json", "list", "paragraph").

        Returns:
            InstructionMetrics.
        """
        n = len(predictions)
        format_scores = []
        relevance_scores = []

        for i, (pred, instr) in enumerate(zip(predictions, instructions)):
            # Format compliance
            if expected_formats and i < len(expected_formats):
                format_scores.append(self._check_format(pred, expected_formats[i]))
            else:
                format_scores.append(1.0)

            # Basic relevance (keyword overlap)
            relevance_scores.append(self._check_relevance(pred, instr))

        metrics = InstructionMetrics(
            format_compliance=sum(format_scores) / n if n else 0.0,
            relevance=sum(relevance_scores) / n if n else 0.0,
            completeness=0.0,  # Requires LLM judge
            num_samples=n,
        )
        metrics.overall_score = (metrics.format_compliance + metrics.relevance) / 2

        logger.info(
            "Instruction eval: format=%.3f, relevance=%.3f, overall=%.3f, n=%d",
            metrics.format_compliance, metrics.relevance,
            metrics.overall_score, metrics.num_samples,
        )
        return metrics

    @staticmethod
    def _check_format(prediction: str, expected_format: str) -> float:
        """Check if output matches expected format."""
        prediction = prediction.strip()
        if expected_format == "json":
            try:
                import json
                json.loads(prediction)
                return 1.0
            except (json.JSONDecodeError, ValueError):
                return 0.0
        elif expected_format == "list":
            lines = prediction.strip().split("\n")
            if any(line.strip().startswith(("-", "*", "1")) for line in lines):
                return 1.0
            return 0.0
        return 1.0  # No format requirement

    @staticmethod
    def _check_relevance(prediction: str, instruction: str) -> float:
        """Basic relevance check via keyword overlap."""
        pred_words = set(prediction.lower().split())
        instr_words = set(instruction.lower().split())
        # Remove stop words
        stop = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at", "to", "for", "of", "and", "or", "this"}
        instr_keywords = instr_words - stop
        if not instr_keywords:
            return 1.0
        overlap = len(pred_words & instr_keywords)
        return min(overlap / max(len(instr_keywords) * 0.3, 1), 1.0)
