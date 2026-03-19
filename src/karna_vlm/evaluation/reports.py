"""
Evaluation report generation.

Produces structured reports combining all evaluation metrics
into a single document for analysis and comparison.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class EvalReport:
    """Comprehensive evaluation report.

    Aggregates metrics from all evaluators into a single
    structured document.
    """

    model_name: str = ""
    model_config: dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    caption_metrics: dict[str, float] = field(default_factory=dict)
    vqa_metrics: dict[str, float] = field(default_factory=dict)
    instruction_metrics: dict[str, float] = field(default_factory=dict)
    grounding_metrics: dict[str, float] = field(default_factory=dict)
    latency_metrics: dict[str, float] = field(default_factory=dict)
    hallucination_rate: float = 0.0
    custom_metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def save(self, path: str | Path) -> None:
        """Save report to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        logger.info("Report saved: %s", path)

    @classmethod
    def load(cls, path: str | Path) -> "EvalReport":
        """Load report from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Evaluation Report: {self.model_name}",
            f"Timestamp: {self.timestamp}",
            "=" * 60,
        ]

        if self.caption_metrics:
            lines.append("\n📝 Caption Quality:")
            for k, v in self.caption_metrics.items():
                lines.append(f"  {k}: {v:.4f}")

        if self.vqa_metrics:
            lines.append("\n❓ VQA Accuracy:")
            for k, v in self.vqa_metrics.items():
                lines.append(f"  {k}: {v:.4f}")

        if self.instruction_metrics:
            lines.append("\n📋 Instruction Following:")
            for k, v in self.instruction_metrics.items():
                lines.append(f"  {k}: {v:.4f}")

        if self.grounding_metrics:
            lines.append("\n🎯 Visual Grounding:")
            for k, v in self.grounding_metrics.items():
                lines.append(f"  {k}: {v:.4f}")

        if self.latency_metrics:
            lines.append("\n⚡ Latency:")
            for k, v in self.latency_metrics.items():
                lines.append(f"  {k}: {v:.1f}")

        if self.hallucination_rate > 0:
            lines.append(f"\n⚠️ Hallucination rate: {self.hallucination_rate:.2%}")

        return "\n".join(lines)

    @staticmethod
    def compare(reports: list["EvalReport"]) -> str:
        """Compare multiple evaluation reports side-by-side.

        Args:
            reports: List of reports to compare.

        Returns:
            Formatted comparison string.
        """
        if not reports:
            return "No reports to compare."

        lines = ["Model Comparison", "=" * 60]

        # Collect all metric categories
        for category in ("caption_metrics", "vqa_metrics", "instruction_metrics", "latency_metrics"):
            category_name = category.replace("_", " ").title()
            all_keys: set[str] = set()
            for r in reports:
                all_keys.update(getattr(r, category, {}).keys())

            if not all_keys:
                continue

            lines.append(f"\n{category_name}:")
            header = "  {:30s}".format("Metric")
            for r in reports:
                header += f" | {r.model_name:>15s}"
            lines.append(header)
            lines.append("  " + "-" * (30 + 18 * len(reports)))

            for key in sorted(all_keys):
                row = f"  {key:30s}"
                for r in reports:
                    val = getattr(r, category, {}).get(key, 0.0)
                    row += f" | {val:>15.4f}"
                lines.append(row)

        return "\n".join(lines)
