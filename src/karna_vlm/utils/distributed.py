"""Distributed training utilities."""

from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def setup_distributed(
    backend: str = "nccl",
    local_rank: Optional[int] = None,
) -> int:
    """Initialize distributed training.

    Args:
        backend: PyTorch distributed backend.
        local_rank: Local rank override.

    Returns:
        Local rank.
    """
    rank = local_rank or int(os.environ.get("LOCAL_RANK", 0))

    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend=backend)
            torch.cuda.set_device(rank)
            logger.info(
                "Distributed training initialized: rank=%d, world=%d",
                rank, torch.distributed.get_world_size(),
            )
    return rank


def is_main_process() -> bool:
    """Check if this is the main process (rank 0)."""
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def get_world_size() -> int:
    """Get world size."""
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()
