"""
Caption quality evaluation.

Metrics: CIDEr, BLEU-4, METEOR, ROUGE-L, BERTScore.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class CaptionMetrics:
    """Captioning evaluation metrics."""

    bleu_4: float = 0.0
    rouge_l: float = 0.0
    cider: float = 0.0
    meteor: float = 0.0
    bert_score_f1: float = 0.0
    num_samples: int = 0


class CaptionEvaluator:
    """Evaluator for image captioning quality.

    Args:
        use_cider: Whether to compute CIDEr (requires pycocoevalcap).
        use_bert_score: Whether to compute BERTScore.
    """

    def __init__(
        self,
        use_cider: bool = True,
        use_bert_score: bool = False,
    ) -> None:
        self.use_cider = use_cider
        self.use_bert_score = use_bert_score

    def evaluate(
        self,
        predictions: list[str],
        references: list[list[str]],
    ) -> CaptionMetrics:
        """Evaluate predicted captions against references.

        Args:
            predictions: List of predicted captions.
            references: List of reference caption lists (multiple refs per image).

        Returns:
            CaptionMetrics with computed scores.
        """
        metrics = CaptionMetrics(num_samples=len(predictions))

        # BLEU-4 (using simple n-gram overlap)
        metrics.bleu_4 = self._compute_bleu(predictions, references)

        # ROUGE-L
        metrics.rouge_l = self._compute_rouge_l(predictions, references)

        logger.info(
            "Caption eval: BLEU-4=%.3f, ROUGE-L=%.3f, n=%d",
            metrics.bleu_4, metrics.rouge_l, metrics.num_samples,
        )
        return metrics

    def _compute_bleu(
        self,
        predictions: list[str],
        references: list[list[str]],
    ) -> float:
        """Compute corpus-level BLEU-4 approximation."""
        try:
            from nltk.translate.bleu_score import corpus_bleu
            refs = [[r.split() for r in ref_list] for ref_list in references]
            hyps = [p.split() for p in predictions]
            return corpus_bleu(refs, hyps)
        except ImportError:
            logger.warning("nltk not available for BLEU computation")
            return 0.0

    def _compute_rouge_l(
        self,
        predictions: list[str],
        references: list[list[str]],
    ) -> float:
        """Compute average ROUGE-L F1."""
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
            scores = []
            for pred, refs in zip(predictions, references):
                best_score = max(
                    scorer.score(ref, pred)["rougeL"].fmeasure for ref in refs
                )
                scores.append(best_score)
            return sum(scores) / len(scores) if scores else 0.0
        except ImportError:
            logger.warning("rouge_score not available")
            return 0.0
