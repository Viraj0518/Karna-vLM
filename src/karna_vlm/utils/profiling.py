"""Profiling utilities for performance analysis."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Generator

import torch

logger = logging.getLogger(__name__)


@contextmanager
def timer(name: str = "operation") -> Generator[dict[str, float], None, None]:
    """Context manager for timing code blocks.

    Usage:
        with timer("encoding") as t:
            output = model.encode(images)
        print(f"Took {t['elapsed_ms']:.1f}ms")
    """
    result: dict[str, float] = {}
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield result
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        result["elapsed_ms"] = elapsed * 1000
        result["elapsed_s"] = elapsed
        logger.debug("%s took %.1fms", name, result["elapsed_ms"])


def get_gpu_memory_info() -> dict[str, float]:
    """Get current GPU memory usage.

    Returns:
        Dict with allocated_mb, reserved_mb, peak_mb.
    """
    if not torch.cuda.is_available():
        return {"allocated_mb": 0, "reserved_mb": 0, "peak_mb": 0}

    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1024 / 1024,
        "reserved_mb": torch.cuda.memory_reserved() / 1024 / 1024,
        "peak_mb": torch.cuda.max_memory_allocated() / 1024 / 1024,
    }


def profile_model_forward(
    model: torch.nn.Module,
    inputs: dict[str, torch.Tensor],
    num_runs: int = 10,
    warmup: int = 3,
) -> dict[str, float]:
    """Profile a model's forward pass.

    Args:
        model: The model to profile.
        inputs: Input tensors dict.
        num_runs: Number of profiling runs.
        warmup: Number of warmup runs.

    Returns:
        Dict with timing statistics.
    """
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            model(**inputs)

    # Profile
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            with timer() as t:
                model(**inputs)
            times.append(t["elapsed_ms"])

    return {
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - sum(times)/len(times))**2 for t in times) / len(times)) ** 0.5,
    }
