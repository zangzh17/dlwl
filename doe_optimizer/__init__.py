"""
DOE Optimizer - Diffractive Optical Element Design and Fabrication Optimization

This package provides tools for designing and optimizing diffractive optical elements (DOEs)
using a layered architecture organized by propagation type.

## Architecture (v2.0)

The package is organized into layers:
- api/: Frontend API schemas and request/response types
- wizard/: Parameter generation from high-level user input
- params/: Structured parameters by propagation type (FFT, SFR, ASM)
- validation/: Parameter validation with structured error messages
- core/: Propagators, loss functions, and optimizers
- evaluation/: Result evaluation and metrics
- visualization/: Plotly-optimized visualization data

## Main Entry Points

Using OptimizationRunner:
    from doe_optimizer import OptimizationRunner, OptimizationRequest
    runner = OptimizationRunner()
    response = runner.run(OptimizationRequest.from_json(user_input))

Using convenience function:
    from doe_optimizer import run_optimization
    response = run_optimization(user_input)
"""

# =============================================================================
# CUDA Configuration - must be done before any torch imports
# =============================================================================
import os
if os.environ.get('FORCE_CPU', ''):
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

__version__ = "2.0.0"

# =============================================================================
# API Layer
# =============================================================================
from .api.request import OptimizationRequest
from .api.response import OptimizationResponse, DOEResultData, MetricsData
from .api.errors import ValidationError, ValidationWarning, ErrorCode, WarningCode

# =============================================================================
# Params Layer
# =============================================================================
from .params.base import PropagationType, PropagatorConfig, PhysicalConstants
from .params.fft_params import FFTParams
from .params.sfr_params import SFRParams
from .params.asm_params import ASMParams
from .params.optimization import LossConfig, LossType, OptimizationConfig, OptMethod

# =============================================================================
# Wizard Layer
# =============================================================================
from .wizard.factory import create_wizard, generate_params
from .wizard.base import BaseWizard, WizardOutput

# =============================================================================
# Validation Layer
# =============================================================================
from .validation.validator import StructuredParamsValidator, validate_params
from .validation.messages import ValidationResult, ValidationMessage

# =============================================================================
# Core Layer
# =============================================================================
from .core.propagator_factory import PropagatorBuilder, build_propagator
from .core.loss import BaseLoss, L1Loss, L2Loss, FocalEfficiencyLoss, create_loss
from .core.propagation import propagation_ASM, propagation_FFT, propagation_SFR

# =============================================================================
# Pipeline Layer
# =============================================================================
from .pipeline.runner import OptimizationRunner, run_optimization
from .pipeline.progress import ProgressReporter, ProgressInfo, CancellationToken

# =============================================================================
# Evaluation Layer
# =============================================================================
from .evaluation.evaluator import (
    Evaluator, evaluate_result,
    FiniteDistanceEvaluation, evaluate_finite_distance_splitter
)
from .evaluation.metrics import EvaluationMetrics

# =============================================================================
# Visualization Layer
# =============================================================================
from .visualization.data import VisualizationData, HeatmapData, BarChartData, ScatterData

# =============================================================================
# All Exports
# =============================================================================

__all__ = [
    # Version
    "__version__",

    # API Layer
    "OptimizationRequest",
    "OptimizationResponse",
    "DOEResultData",
    "MetricsData",
    "ValidationError",
    "ValidationWarning",
    "ErrorCode",
    "WarningCode",

    # Params Layer
    "PropagationType",
    "PropagatorConfig",
    "PhysicalConstants",
    "FFTParams",
    "SFRParams",
    "ASMParams",
    "LossConfig",
    "LossType",
    "OptimizationConfig",
    "OptMethod",

    # Wizard Layer
    "create_wizard",
    "generate_params",
    "BaseWizard",
    "WizardOutput",

    # Validation Layer
    "StructuredParamsValidator",
    "validate_params",
    "ValidationResult",
    "ValidationMessage",

    # Core Layer
    "PropagatorBuilder",
    "build_propagator",
    "BaseLoss",
    "L1Loss",
    "L2Loss",
    "FocalEfficiencyLoss",
    "create_loss",
    "propagation_ASM",
    "propagation_FFT",
    "propagation_SFR",

    # Pipeline Layer
    "OptimizationRunner",
    "run_optimization",
    "ProgressReporter",
    "ProgressInfo",
    "CancellationToken",

    # Evaluation Layer
    "Evaluator",
    "evaluate_result",
    "EvaluationMetrics",
    "FiniteDistanceEvaluation",
    "evaluate_finite_distance_splitter",

    # Visualization Layer
    "VisualizationData",
    "HeatmapData",
    "BarChartData",
    "ScatterData",
]
