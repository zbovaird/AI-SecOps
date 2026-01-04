"""
Benchmark module for cross-model comparison.

Phase 4: Cross-Model Benchmarking Harness
"""

from .harness import (
    BenchmarkHarness,
    BenchmarkConfig,
    BenchmarkResult,
    ModelScore,
    run_benchmark,
)

__all__ = [
    "BenchmarkHarness",
    "BenchmarkConfig",
    "BenchmarkResult",
    "ModelScore",
    "run_benchmark",
]





