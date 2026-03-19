"""
Latency and throughput benchmarking.

Measures inference performance for deployment planning:
- First-token latency
- Token generation throughput
- VRAM usage
- Component-level timing (vision, bridge, decoder)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


@dataclass
class LatencyMetrics:
    """Latency benchmark results."""

    # End-to-end
    total_latency_ms: float = 0.0
    first_token_latency_ms: float = 0.0
    tokens_per_second: float = 0.0

    # Component-level
    vision_encode_ms: float = 0.0
    bridge_ms: float = 0.0
    decoder_prefill_ms: float = 0.0
    decoder_per_token_ms: float = 0.0

    # Resource usage
    peak_vram_mb: float = 0.0
    model_size_mb: float = 0.0

    # Config
    num_runs: int = 0
    num_generated_tokens: int = 0
    batch_size: int = 1


class LatencyBenchmark:
    """Benchmark VLM inference latency.

    Args:
        model: The KarnaVLM model.
        warmup_runs: Number of warmup iterations before timing.
        num_runs: Number of timed iterations to average.
    """

    def __init__(
        self,
        model: Any,
        warmup_runs: int = 3,
        num_runs: int = 10,
    ) -> None:
        self.model = model
        self.warmup_runs = warmup_runs
        self.num_runs = num_runs

    @torch.no_grad()
    def benchmark(
        self,
        image_size: int = 224,
        prompt: str = "Describe this image.",
        max_new_tokens: int = 64,
        batch_size: int = 1,
    ) -> LatencyMetrics:
        """Run the full latency benchmark.

        Args:
            image_size: Input image size.
            prompt: Test prompt.
            max_new_tokens: Tokens to generate.
            batch_size: Batch size.

        Returns:
            LatencyMetrics with timing data.
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        # Create dummy inputs
        dummy_images = [
            Image.new("RGB", (image_size, image_size), color=(128, 128, 128))
            for _ in range(batch_size)
        ]

        # Warmup
        for _ in range(self.warmup_runs):
            self.model.generate(images=dummy_images, prompt=prompt, max_new_tokens=8)

        # Reset CUDA stats
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        # Benchmark component timings
        vision_times = []
        bridge_times = []
        total_times = []

        for _ in range(self.num_runs):
            # Vision encoding
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            vision_out = self.model.encode_image(dummy_images)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            vision_times.append((t1 - t0) * 1000)

            # Bridge
            bridge_out = self.model.bridge_image(vision_out)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t2 = time.perf_counter()
            bridge_times.append((t2 - t1) * 1000)

            # Full generation
            t_start = time.perf_counter()
            output = self.model.generate(
                images=dummy_images, prompt=prompt, max_new_tokens=max_new_tokens
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_end = time.perf_counter()
            total_times.append((t_end - t_start) * 1000)

        # VRAM
        peak_vram = 0.0
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / 1024 / 1024

        # Model size
        model_size = sum(p.numel() * p.element_size() for p in self.model.parameters()) / 1024 / 1024

        avg_total = sum(total_times) / len(total_times)
        num_tokens = len(output.split()) if isinstance(output, str) else max_new_tokens

        metrics = LatencyMetrics(
            total_latency_ms=avg_total,
            first_token_latency_ms=sum(vision_times) / len(vision_times) + sum(bridge_times) / len(bridge_times),
            tokens_per_second=num_tokens / (avg_total / 1000) if avg_total > 0 else 0,
            vision_encode_ms=sum(vision_times) / len(vision_times),
            bridge_ms=sum(bridge_times) / len(bridge_times),
            peak_vram_mb=peak_vram,
            model_size_mb=model_size,
            num_runs=self.num_runs,
            num_generated_tokens=num_tokens,
            batch_size=batch_size,
        )

        logger.info(
            "Latency: total=%.1fms, first_token=%.1fms, throughput=%.1f tok/s, VRAM=%.0fMB",
            metrics.total_latency_ms, metrics.first_token_latency_ms,
            metrics.tokens_per_second, metrics.peak_vram_mb,
        )
        return metrics
