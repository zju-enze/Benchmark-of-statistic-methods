"""Evaluation framework for benchmarks."""

from .cross_validation import cross_validate
from .evaluate import evaluate
from .benchmark import run_benchmark_synthetic, run_benchmark_real
from .upload import upload_results, upload_all_results, setup_and_upload

__all__ = [
    "cross_validate",
    "evaluate",
    "run_benchmark_synthetic",
    "run_benchmark_real",
    "upload_results",
    "upload_all_results",
    "setup_and_upload",
]
