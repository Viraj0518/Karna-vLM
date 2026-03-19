"""
Ablation study utilities.

Systematic comparison of model variants (bridge types, query counts,
LoRA ranks, etc.) with standardized reporting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


@dataclass
class AblationResult:
    """Result of a single ablation experiment."""

    name: str
    config_diff: dict[str, Any]  # What changed from baseline
    metrics: dict[str, float]
    notes: str = ""


@dataclass
class AblationStudy:
    """Collection of ablation results with comparison."""

    study_name: str
    baseline_name: str = ""
    results: list[AblationResult] = field(default_factory=list)

    def add_result(self, result: AblationResult) -> None:
        self.results.append(result)

    def get_comparison_table(self) -> list[dict[str, Any]]:
        """Generate a comparison table across all ablation results.

        Returns:
            List of dicts with name, config changes, and all metric values.
        """
        if not self.results:
            return []

        # Collect all metric keys
        all_metrics = set()
        for r in self.results:
            all_metrics.update(r.metrics.keys())

        table = []
        baseline_metrics = {}
        if self.baseline_name:
            for r in self.results:
                if r.name == self.baseline_name:
                    baseline_metrics = r.metrics
                    break

        for r in self.results:
            row = {
                "name": r.name,
                "config": str(r.config_diff),
            }
            for metric in sorted(all_metrics):
                value = r.metrics.get(metric, 0.0)
                row[metric] = value
                if baseline_metrics and metric in baseline_metrics:
                    base_val = baseline_metrics[metric]
                    diff = value - base_val
                    row[f"{metric}_delta"] = diff
            table.append(row)

        return table

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [f"Ablation Study: {self.study_name}", "=" * 60]
        table = self.get_comparison_table()
        for row in table:
            lines.append(f"\n{row['name']}:")
            lines.append(f"  Config: {row['config']}")
            for k, v in row.items():
                if k not in ("name", "config") and not k.endswith("_delta"):
                    delta_key = f"{k}_delta"
                    delta_str = ""
                    if delta_key in row:
                        d = row[delta_key]
                        delta_str = f" ({'+'if d >= 0 else ''}{d:.4f})"
                    lines.append(f"  {k}: {v:.4f}{delta_str}")
        return "\n".join(lines)


def run_bridge_ablation(
    model_factory: Callable,
    eval_fn: Callable,
    bridge_types: list[str],
    base_config: dict[str, Any],
) -> AblationStudy:
    """Run ablation study across bridge types.

    Args:
        model_factory: Function(config_dict) -> model.
        eval_fn: Function(model) -> metrics dict.
        bridge_types: List of bridge type names to compare.
        base_config: Base model configuration.

    Returns:
        AblationStudy with results for each bridge type.
    """
    study = AblationStudy(study_name="Bridge Type Ablation", baseline_name=bridge_types[0])

    for bridge_type in bridge_types:
        config = {**base_config, "bridge_type": bridge_type}
        model = model_factory(config)
        metrics = eval_fn(model)

        study.add_result(AblationResult(
            name=f"bridge={bridge_type}",
            config_diff={"bridge_type": bridge_type},
            metrics=metrics,
        ))
        logger.info("Ablation %s: %s", bridge_type, metrics)

    return study
