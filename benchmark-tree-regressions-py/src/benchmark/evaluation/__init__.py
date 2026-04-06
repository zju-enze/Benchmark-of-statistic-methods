"""Evaluation framework for benchmarks."""

from .cross_validation import cross_validate
from .evaluate import evaluate
from .benchmark import run_benchmark_synthetic, run_benchmark_real

__all__ = [
    "cross_validate",
    "evaluate",
    "run_benchmark_synthetic",
    "run_benchmark_real",
]
