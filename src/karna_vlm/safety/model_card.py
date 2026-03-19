"""
Model card generation for Karna VLM.

Produces standardized model cards following the Model Card framework
for responsible AI documentation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class ModelCard:
    """Model card for a Karna VLM variant.

    Documents model details, training data, evaluation results,
    limitations, and ethical considerations.
    """

    # Model details
    model_name: str = "Karna VLM"
    model_version: str = "0.1.0"
    model_type: str = "Vision-Language Model"
    architecture_summary: str = ""

    # Training
    training_data: list[str] = field(default_factory=list)
    training_compute: str = ""
    training_stages: list[str] = field(default_factory=list)

    # Evaluation
    eval_results: dict[str, float] = field(default_factory=dict)
    eval_datasets: list[str] = field(default_factory=list)

    # Intended use
    primary_uses: list[str] = field(default_factory=list)
    out_of_scope_uses: list[str] = field(default_factory=list)

    # Limitations
    known_limitations: list[str] = field(default_factory=list)
    failure_modes: list[str] = field(default_factory=list)

    # Ethical considerations
    ethical_considerations: list[str] = field(default_factory=list)
    bias_notes: list[str] = field(default_factory=list)

    # License
    license: str = "Apache-2.0"
    citation: str = ""

    # Extra
    extra: dict[str, Any] = field(default_factory=dict)

    def generate_markdown(self) -> str:
        """Generate a Markdown model card.

        Returns:
            Formatted Markdown string.
        """
        sections = []

        sections.append(f"# Model Card: {self.model_name} v{self.model_version}\n")

        # Model Details
        sections.append("## Model Details\n")
        sections.append(f"- **Type:** {self.model_type}")
        sections.append(f"- **Version:** {self.model_version}")
        sections.append(f"- **License:** {self.license}")
        if self.architecture_summary:
            sections.append(f"- **Architecture:** {self.architecture_summary}")
        sections.append("")

        # Training
        if self.training_data or self.training_stages:
            sections.append("## Training\n")
            if self.training_stages:
                sections.append("### Training Stages")
                for stage in self.training_stages:
                    sections.append(f"- {stage}")
                sections.append("")
            if self.training_data:
                sections.append("### Training Data")
                for dataset in self.training_data:
                    sections.append(f"- {dataset}")
                sections.append("")
            if self.training_compute:
                sections.append(f"**Compute:** {self.training_compute}\n")

        # Evaluation
        if self.eval_results:
            sections.append("## Evaluation Results\n")
            sections.append("| Metric | Score |")
            sections.append("|--------|-------|")
            for metric, score in self.eval_results.items():
                sections.append(f"| {metric} | {score:.4f} |")
            sections.append("")

        # Intended Use
        sections.append("## Intended Use\n")
        if self.primary_uses:
            sections.append("### Primary Uses")
            for use in self.primary_uses:
                sections.append(f"- {use}")
            sections.append("")
        if self.out_of_scope_uses:
            sections.append("### Out of Scope")
            for use in self.out_of_scope_uses:
                sections.append(f"- ⚠️ {use}")
            sections.append("")

        # Limitations
        if self.known_limitations:
            sections.append("## Limitations\n")
            for lim in self.known_limitations:
                sections.append(f"- {lim}")
            sections.append("")

        # Ethics
        if self.ethical_considerations:
            sections.append("## Ethical Considerations\n")
            for consideration in self.ethical_considerations:
                sections.append(f"- {consideration}")
            sections.append("")

        # Citation
        if self.citation:
            sections.append("## Citation\n")
            sections.append(f"```\n{self.citation}\n```\n")

        return "\n".join(sections)

    def save(self, path: str | Path) -> None:
        """Save model card as Markdown."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.generate_markdown(), encoding="utf-8")
        logger.info("Model card saved: %s", path)


def generate_default_card(model_config: Any) -> ModelCard:
    """Generate a default model card from a KarnaVLMConfig.

    Args:
        model_config: KarnaVLMConfig instance.

    Returns:
        Pre-filled ModelCard.
    """
    return ModelCard(
        model_name=f"Karna VLM ({model_config.model_family})",
        architecture_summary=(
            f"Vision: {model_config.vision_backend} ({model_config.vision_model}) | "
            f"Bridge: {model_config.bridge_type} (dim={model_config.bridge_dim}, "
            f"queries={model_config.bridge_num_queries}) | "
            f"Decoder: {model_config.decoder_model}"
        ),
        training_stages=[
            "Stage 1: Bootstrap alignment (bridge only, captioning)",
            "Stage 2: Multitask instruction tuning (bridge + decoder LoRA)",
            "Stage 3: Domain specialization (domain adapter)",
            "Stage 4: Distillation & compression",
        ],
        primary_uses=[
            "Image captioning and description",
            "Visual question answering",
            "Multimodal instruction following",
            "Document and image understanding",
            "Domain-specific visual analysis (with adapters)",
        ],
        out_of_scope_uses=[
            "Medical diagnosis without expert review",
            "Autonomous decision-making in safety-critical systems",
            "Generating deceptive or harmful content",
        ],
        known_limitations=[
            "May hallucinate details not present in the image",
            "Performance degrades on very small or low-quality images",
            "Limited to single-image input (no video)",
            "May reflect biases present in training data",
        ],
        ethical_considerations=[
            "Training data provenance should be verified for commercial use",
            "Outputs should be validated by humans for critical applications",
            "Model should not be used to extract private information from images",
        ],
    )
