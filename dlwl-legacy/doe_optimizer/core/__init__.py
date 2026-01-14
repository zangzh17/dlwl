"""Core modules for DOE optimization.

This module provides:
- PropagatorBuilder: Independent propagator creation
- Loss functions: Pluggable L1, L2, FocalEfficiency losses
- Propagation functions: ASM, FFT, SFR algorithms
"""

from .propagator_factory import PropagatorBuilder, build_propagator
from .loss import BaseLoss, L1Loss, L2Loss, FocalEfficiencyLoss, create_loss
from .propagation import propagation_ASM, propagation_FFT, propagation_SFR

__all__ = [
    # Propagator
    "PropagatorBuilder",
    "build_propagator",
    # Loss functions
    "BaseLoss",
    "L1Loss",
    "L2Loss",
    "FocalEfficiencyLoss",
    "create_loss",
    # Propagation algorithms
    "propagation_ASM",
    "propagation_FFT",
    "propagation_SFR",
]
