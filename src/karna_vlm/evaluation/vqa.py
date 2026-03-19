"""
Visual Question Answering evaluation.

Metrics: Exact match, relaxed accuracy, VQA accuracy (soft matching).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class VQAMetrics:
    """VQA evaluation metrics."""

    exact_match: float = 0.0
    relaxed_accuracy: float = 0.0
    vqa_accuracy: float = 0.0
    num_samples: int = 0


class VQAEvaluator:
    """Evaluator for VQA tasks.

    Supports multiple accuracy modes:
    - Exact match (case-insensitive string equality)
    - Relaxed accuracy (answer contained in prediction)
    - VQA accuracy (soft matching with human annotation weighting)
    """

    def __init__(self) -> None:
        pass

    def evaluate(
        self,
        predictions: list[str],
        ground_truths: list[str | list[str]],
    ) -> VQAMetrics:
        """Evaluate VQA predictions.

        Args:
            predictions: Predicted answers.
            ground_truths: Ground truth answers (str or list of acceptable answers).

        Returns:
            VQAMetrics with computed scores.
        """
        n = len(predictions)
        exact = 0
        relaxed = 0
        vqa_scores = []

        for pred, gt in zip(predictions, ground_truths):
            pred_clean = self._normalize(pred)

            if isinstance(gt, str):
                gt_list = [gt]
            else:
                gt_list = gt

            gt_clean = [self._normalize(g) for g in gt_list]

            # Exact match
            if pred_clean in gt_clean:
                exact += 1

            # Relaxed accuracy
            if any(g in pred_clean or pred_clean in g for g in gt_clean):
                relaxed += 1

            # VQA accuracy (min(#humans_that_gave_answer / 3, 1))
            match_count = sum(1 for g in gt_clean if g == pred_clean)
            vqa_scores.append(min(match_count / 3.0, 1.0))

        metrics = VQAMetrics(
            exact_match=exact / n if n > 0 else 0.0,
            relaxed_accuracy=relaxed / n if n > 0 else 0.0,
            vqa_accuracy=sum(vqa_scores) / n if n > 0 else 0.0,
            num_samples=n,
        )

        logger.info(
            "VQA eval: exact=%.3f, relaxed=%.3f, vqa_acc=%.3f, n=%d",
            metrics.exact_match, metrics.relaxed_accuracy,
            metrics.vqa_accuracy, metrics.num_samples,
        )
        return metrics

    @staticmethod
    def _normalize(text: str) -> str:
        """Normalize answer text for comparison."""
        text = text.lower().strip()
        # Remove articles
        text = re.sub(r"\b(a|an|the)\b", " ", text)
        # Remove punctuation
        text = re.sub(r"[^\w\s]", "", text)
        # Collapse whitespace
        text = " ".join(text.split())
        return text
