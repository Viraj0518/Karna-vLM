"""
Dataset implementations for Karna VLM training and evaluation.

Supports loading from local files, HuggingFace datasets, and
custom manifest-based datasets.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from karna_vlm.data.schemas import VLMSample, TaskType
from karna_vlm.data.templates import PromptTemplate, TEMPLATES

logger = logging.getLogger(__name__)


class VLMDataset(Dataset):
    """Unified VLM dataset.

    Loads samples from JSONL files with the VLMSample schema and
    applies prompt templates and image loading.

    Args:
        data_path: Path to JSONL data file or directory.
        image_root: Root directory for resolving relative image paths.
        template: Prompt template to apply.
        transform: Optional image transform (applied after loading).
        max_samples: Optional limit on number of samples.
        task_filter: Optional task type filter.
    """

    def __init__(
        self,
        data_path: str | Path,
        image_root: str | Path = "",
        template: Optional[PromptTemplate] = None,
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
        task_filter: Optional[TaskType] = None,
    ) -> None:
        self.data_path = Path(data_path)
        self.image_root = Path(image_root) if image_root else None
        self.template = template or TEMPLATES.get("default")
        self.transform = transform
        self.samples: list[VLMSample] = []

        self._load_data(max_samples, task_filter)
        logger.info("Loaded %d samples from %s", len(self.samples), self.data_path)

    def _load_data(
        self,
        max_samples: Optional[int],
        task_filter: Optional[TaskType],
    ) -> None:
        """Load samples from JSONL file(s)."""
        paths = []
        if self.data_path.is_file():
            paths = [self.data_path]
        elif self.data_path.is_dir():
            paths = sorted(self.data_path.glob("*.jsonl"))

        for path in paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if max_samples and len(self.samples) >= max_samples:
                        break
                    data = json.loads(line.strip())
                    sample = VLMSample(
                        image_path=data["image_path"],
                        conversations=data.get("conversations", []),
                        task_type=TaskType(data.get("task_type", "instruction")),
                        metadata=data.get("metadata", {}),
                        image_id=data.get("image_id"),
                        bbox=data.get("bbox"),
                        source_dataset=data.get("source_dataset", ""),
                    )
                    if task_filter and sample.task_type != task_filter:
                        continue
                    self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a processed sample.

        Returns:
            Dict with keys: image (PIL), prompt (str), response (str),
            task_type (str), metadata (dict).
        """
        sample = self.samples[idx]

        # Load image
        image_path = sample.image_path
        if self.image_root and not Path(image_path).is_absolute():
            image_path = str(self.image_root / image_path)

        try:
            image = Image.open(image_path).convert("RGB")
        except (FileNotFoundError, Exception) as e:
            logger.warning("Failed to load image %s: %s", image_path, e)
            # Return a small placeholder
            image = Image.new("RGB", (224, 224), color=(128, 128, 128))

        if self.transform:
            image = self.transform(image)

        # Apply template
        prompt = sample.get_prompt()
        response = sample.get_response()

        if self.template:
            prompt = self.template.format_prompt(prompt, task_type=sample.task_type)

        return {
            "image": image,
            "prompt": prompt,
            "response": response,
            "task_type": sample.task_type.value,
            "metadata": sample.metadata,
        }


class HFVLMDataset(Dataset):
    """Wrapper for HuggingFace datasets with VLM-compatible format.

    Args:
        dataset_name: HuggingFace dataset identifier.
        split: Dataset split (train/val/test).
        image_column: Column containing images.
        text_column: Column containing text (question/prompt).
        label_column: Column containing labels/answers.
        max_samples: Optional limit.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        image_column: str = "image",
        text_column: str = "question",
        label_column: str = "answer",
        max_samples: Optional[int] = None,
    ) -> None:
        from datasets import load_dataset

        self.dataset = load_dataset(dataset_name, split=split)
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        self.image_column = image_column
        self.text_column = text_column
        self.label_column = label_column

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.dataset[idx]
        image = item[self.image_column]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")

        return {
            "image": image,
            "prompt": str(item.get(self.text_column, "")),
            "response": str(item.get(self.label_column, "")),
            "task_type": "vqa",
            "metadata": {},
        }
