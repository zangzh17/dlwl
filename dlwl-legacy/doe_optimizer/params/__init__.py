"""
Structured parameters organized by propagation type.

This module defines the three types of structured parameters:
- FFTParams: Type A - k-space / infinite distance
- SFRParams: Type B - large target face finite distance
- ASMParams: Type C - small target face finite distance

These replace the old DOEConfig system with a cleaner separation
based on the underlying physics (propagation algorithm).
"""

from .base import (
    PropagationType,
    PropagatorConfig,
    PhysicalConstants,
    StructuredParams,
)
from .fft_params import FFTParams
from .sfr_params import SFRParams
from .asm_params import ASMParams
from .optimization import (
    LossConfig,
    LossType,
    OptimizationConfig,
    OptMethod,
)

__all__ = [
    # Base types
    'PropagationType',
    'PropagatorConfig',
    'PhysicalConstants',
    'StructuredParams',
    # Parameter types
    'FFTParams',
    'SFRParams',
    'ASMParams',
    # Optimization config
    'LossConfig',
    'LossType',
    'OptimizationConfig',
    'OptMethod',
]
