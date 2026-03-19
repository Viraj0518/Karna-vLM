"""
Unified data schemas for the Karna VLM data pipeline.

Every dataset fed into the training or evaluation pipeline must
conform to these schemas. This ensures provenance tracking,
licensing compliance, and consistent processing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class TaskType(str, Enum):
    """Supported VLM task types."""
    CAPTION = "caption"
    VQA = "vqa"
    INSTRUCTION = "instruction"
    CHAT = "chat"
    GROUNDING = "grounding"
    OCR = "ocr"
    CLASSIFICATION = "classification"
    STRUCTURED_EXTRACTION = "structured_extraction"


class License(str, Enum):
    """Common dataset licenses."""
    CC_BY_4 = "CC-BY-4.0"
    CC_BY_SA_4 = "CC-BY-SA-4.0"
    CC_BY_NC_4 = "CC-BY-NC-4.0"
    APACHE_2 = "Apache-2.0"
    MIT = "MIT"
    PROPRIETARY = "proprietary"
    RESEARCH_ONLY = "research-only"
    UNKNOWN = "unknown"


@dataclass
class VLMSample:
    """A single training/evaluation sample.

    Attributes:
        image_path: Path or URL to the image.
        conversations: List of conversation turns [{role, content}].
        task_type: Type of task this sample represents.
        metadata: Additional metadata (source, split, etc.).
        image_id: Optional unique image identifier.
        bbox: Optional bounding boxes for grounding tasks.
        source_dataset: Name of the source dataset.
    """

    image_path: str
    conversations: list[dict[str, str]]
    task_type: TaskType = TaskType.INSTRUCTION
    metadata: dict[str, Any] = field(default_factory=dict)
    image_id: Optional[str] = None
    bbox: Optional[list[list[float]]] = None  # [[x1,y1,x2,y2], ...]
    source_dataset: str = ""

    def get_prompt(self) -> str:
        """Extract the user prompt from conversations."""
        for turn in self.conversations:
            if turn.get("role") in ("user", "human"):
                return turn["content"]
        return ""

    def get_response(self) -> str:
        """Extract the assistant response from conversations."""
        for turn in self.conversations:
            if turn.get("role") in ("assistant", "gpt"):
                return turn["content"]
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_path": self.image_path,
            "conversations": self.conversations,
            "task_type": self.task_type.value,
            "metadata": self.metadata,
            "image_id": self.image_id,
            "bbox": self.bbox,
            "source_dataset": self.source_dataset,
        }


@dataclass
class DatasetManifest:
    """Manifest for a dataset — metadata, provenance, and licensing.

    Every dataset used in training or evaluation must have a manifest
    for governance and audit purposes.

    Attributes:
        name: Human-readable dataset name.
        version: Dataset version string.
        license: License type.
        source_url: Original source URL.
        num_samples: Total number of samples.
        task_types: Task types present in this dataset.
        description: Brief description.
        citation: BibTeX or text citation.
        provenance: How the data was collected/created.
        quality_notes: Known quality issues or caveats.
    """

    name: str
    version: str = "1.0.0"
    license: License = License.UNKNOWN
    source_url: str = ""
    num_samples: int = 0
    task_types: list[TaskType] = field(default_factory=list)
    description: str = ""
    citation: str = ""
    provenance: str = ""
    quality_notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def is_commercial_safe(self) -> bool:
        """Check if this dataset's license allows commercial use."""
        return self.license in (
            License.CC_BY_4,
            License.CC_BY_SA_4,
            License.APACHE_2,
            License.MIT,
            License.PROPRIETARY,
        )
