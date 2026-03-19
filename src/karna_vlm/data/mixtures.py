"""
Dataset mixture system for multi-task training.

Supports weighted sampling from multiple datasets, allowing
control over task proportions during training.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from torch.utils.data import Dataset, Sampler

logger = logging.getLogger(__name__)


@dataclass
class MixtureComponent:
    """A single component in a dataset mixture.

    Attributes:
        dataset: The dataset instance.
        weight: Sampling weight (relative to other components).
        name: Human-readable name.
        task_type: Primary task type.
    """

    dataset: Dataset
    weight: float = 1.0
    name: str = ""
    task_type: str = ""


class DatasetMixture(Dataset):
    """Weighted mixture of multiple datasets.

    Samples are drawn from component datasets proportional to their
    weights, enabling controlled multi-task training.

    Args:
        components: List of MixtureComponent objects.
        total_samples: Virtual size of the mixture per epoch.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        components: list[MixtureComponent],
        total_samples: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.components = components
        self.seed = seed
        self.rng = random.Random(seed)

        # Normalize weights
        total_weight = sum(c.weight for c in components)
        self.normalized_weights = [c.weight / total_weight for c in components]

        # Compute total samples
        if total_samples is not None:
            self._total = total_samples
        else:
            self._total = sum(len(c.dataset) for c in components)

        # Pre-compute sample allocation
        self._allocations = self._compute_allocations()

        logger.info(
            "Mixture created: %d components, %d total samples",
            len(components),
            self._total,
        )
        for i, c in enumerate(components):
            logger.info(
                "  [%d] %s: weight=%.2f, size=%d, effective=%.1f%%",
                i,
                c.name or f"dataset_{i}",
                c.weight,
                len(c.dataset),
                self.normalized_weights[i] * 100,
            )

    def _compute_allocations(self) -> list[tuple[int, int]]:
        """Compute (dataset_idx, sample_idx) for each virtual index."""
        allocations = []
        for idx in range(self._total):
            # Weighted random selection
            r = self.rng.random()
            cumulative = 0.0
            dataset_idx = 0
            for i, w in enumerate(self.normalized_weights):
                cumulative += w
                if r <= cumulative:
                    dataset_idx = i
                    break

            # Random sample within the selected dataset
            sample_idx = self.rng.randint(0, len(self.components[dataset_idx].dataset) - 1)
            allocations.append((dataset_idx, sample_idx))

        return allocations

    def __len__(self) -> int:
        return self._total

    def __getitem__(self, idx: int) -> dict[str, Any]:
        dataset_idx, sample_idx = self._allocations[idx]
        item = self.components[dataset_idx].dataset[sample_idx]
        # Wrap non-dict items (e.g., TensorDataset returns tuples)
        if isinstance(item, dict):
            item["mixture_source"] = self.components[dataset_idx].name
            return item
        return {
            "data": item,
            "mixture_source": self.components[dataset_idx].name,
        }


class BalancedMixtureSampler(Sampler):
    """Sampler that ensures balanced representation across mixture components.

    Cycles through datasets in proportion to their weights,
    ensuring no dataset is starved in any training window.
    """

    def __init__(
        self,
        mixture: DatasetMixture,
        batch_size: int = 32,
        seed: int = 42,
    ) -> None:
        self.mixture = mixture
        self.batch_size = batch_size
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        indices = list(range(len(self.mixture)))
        rng.shuffle(indices)
        return iter(indices)

    def __len__(self) -> int:
        return len(self.mixture)
