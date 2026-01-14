"""
Evaluation module for DOE optimization results.

This module provides:
- Evaluator class for computing metrics
- EvaluationMetrics dataclass
- Specialized evaluation for different DOE types
- FiniteDistanceEvaluation for SFR-based evaluation with Airy disk integration
"""

from .metrics import EvaluationMetrics
from .evaluator import (
    Evaluator,
    evaluate_result,
    FiniteDistanceEvaluation,
    evaluate_finite_distance_splitter,
)

__all__ = [
    'EvaluationMetrics',
    'Evaluator',
    'evaluate_result',
    'FiniteDistanceEvaluation',
    'evaluate_finite_distance_splitter',
]
