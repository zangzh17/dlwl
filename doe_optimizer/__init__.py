"""
DOE Optimizer - Diffractive Optical Element Design and Fabrication Optimization

This package provides tools for designing and optimizing diffractive optical elements (DOEs)
using a two-step optimization approach:
1. Phase/Height optimization (traditional DOE design)
2. Fabrication optimization (OPE correction for laser direct writing)

Main entry point: optimize_doe()
"""

from .core.config import (
    DOEType,
    PropModel,
    SplitterMode,
    FiniteDistanceStrategy,
    PhysicalParams,
    DeviceParams,
    OptimizationParams,
    FabricationCalibration,
    TargetParams,
    DOEConfig,
    MAX_OPTIMIZATION_RESOLUTION,
)
from .pipeline.two_step import optimize_doe, DOEResult
from .pipeline.evaluation import EvaluationMetrics, evaluate_result

__version__ = "0.1.0"
__all__ = [
    "DOEType",
    "PropModel",
    "SplitterMode",
    "FiniteDistanceStrategy",
    "PhysicalParams",
    "DeviceParams",
    "OptimizationParams",
    "FabricationCalibration",
    "TargetParams",
    "DOEConfig",
    "optimize_doe",
    "DOEResult",
    "EvaluationMetrics",
    "evaluate_result",
    "MAX_OPTIMIZATION_RESOLUTION",
]
