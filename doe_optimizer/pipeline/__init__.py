"""DOE optimization pipeline.

This module provides:
- OptimizationRunner: Unified entry point for DOE optimization
- run_optimization: Convenience function for optimization
- ProgressReporter: Progress callback support with cancellation
"""

from .runner import OptimizationRunner, run_optimization
from .progress import ProgressReporter, ProgressInfo, CancellationToken

__all__ = [
    "OptimizationRunner",
    "run_optimization",
    "ProgressReporter",
    "ProgressInfo",
    "CancellationToken",
]
