"""
Visual grounding evaluation.

Metrics: IoU, precision@IoU thresholds, mAP.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)


@dataclass
class GroundingMetrics:
    """Visual grounding evaluation metrics."""

    mean_iou: float = 0.0
    precision_at_50: float = 0.0  # Precision @ IoU >= 0.5
    precision_at_75: float = 0.0  # Precision @ IoU >= 0.75
    num_samples: int = 0


class GroundingEvaluator:
    """Evaluator for visual grounding tasks."""

    def evaluate(
        self,
        pred_boxes: list[list[float]],
        gt_boxes: list[list[float]],
    ) -> GroundingMetrics:
        """Evaluate grounding predictions.

        Args:
            pred_boxes: Predicted boxes [[x1,y1,x2,y2], ...] normalized 0-1.
            gt_boxes: Ground truth boxes.

        Returns:
            GroundingMetrics.
        """
        n = len(pred_boxes)
        ious = []

        for pred, gt in zip(pred_boxes, gt_boxes):
            iou = self._compute_iou(pred, gt)
            ious.append(iou)

        metrics = GroundingMetrics(
            mean_iou=sum(ious) / n if n else 0.0,
            precision_at_50=sum(1 for iou in ious if iou >= 0.5) / n if n else 0.0,
            precision_at_75=sum(1 for iou in ious if iou >= 0.75) / n if n else 0.0,
            num_samples=n,
        )

        logger.info(
            "Grounding eval: mIoU=%.3f, P@50=%.3f, P@75=%.3f, n=%d",
            metrics.mean_iou, metrics.precision_at_50,
            metrics.precision_at_75, metrics.num_samples,
        )
        return metrics

    @staticmethod
    def _compute_iou(box1: list[float], box2: list[float]) -> float:
        """Compute IoU between two boxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
        area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
        union = area1 + area2 - inter

        return inter / union if union > 0 else 0.0
