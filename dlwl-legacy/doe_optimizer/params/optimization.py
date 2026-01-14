"""
Optimization configuration parameters.

This module defines configuration for:
- Loss functions (type, weights, ROI)
- Optimization algorithms (method, learning rate, iterations)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Literal
from enum import Enum
import numpy as np


class LossType(Enum):
    """Available loss function types."""
    L1 = "L1"                        # Mean absolute error
    L2 = "L2"                        # Mean squared error
    FOCAL_EFFICIENCY = "focal_efficiency"  # Maximize energy in Airy disk
    FOCAL_UNIFORMITY = "focal_uniformity"  # Minimize efficiency variance
    COMPOSITE = "composite"          # Weighted combination


class OptMethod(Enum):
    """Available optimization methods."""
    SGD = "SGD"    # Stochastic Gradient Descent (with Adam)
    GS = "GS"      # Gerchberg-Saxton iterative
    BS = "BS"      # Binary Search


@dataclass
class LossConfig:
    """Loss function configuration.

    Attributes:
        loss_type: Type of loss function
        weights: Weights for composite loss (e.g., {'L2': 1.0, 'focal_efficiency': 0.5})
        roi_enabled: Whether to use ROI-based loss
        roi_mask: ROI mask array (optional, set later)
        focal_params: Parameters for focal efficiency loss
            - spot_positions: List of (y, x) spot positions
            - airy_radius: Airy disk radius in pixels

    Example:
        # Simple L2 loss
        config = LossConfig(loss_type=LossType.L2)

        # Composite loss for splitters
        config = LossConfig(
            loss_type=LossType.COMPOSITE,
            weights={'L2': 0.5, 'focal_uniformity': 0.5}
        )
    """
    loss_type: LossType = LossType.L2
    weights: Optional[Dict[str, float]] = None
    roi_enabled: bool = False
    roi_mask: Optional[np.ndarray] = field(default=None, repr=False)
    focal_params: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate configuration."""
        if self.loss_type == LossType.COMPOSITE and not self.weights:
            raise ValueError("COMPOSITE loss requires weights")

        if self.loss_type == LossType.FOCAL_EFFICIENCY and not self.focal_params:
            # Will be set by wizard
            pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding roi_mask array)."""
        result = {
            'loss_type': self.loss_type.value,
            'roi_enabled': self.roi_enabled,
        }
        if self.weights:
            result['weights'] = self.weights
        if self.focal_params:
            result['focal_params'] = self.focal_params
        return result


@dataclass
class OptimizationConfig:
    """Complete optimization configuration.

    Attributes:
        phase_method: Algorithm for phase optimization
        phase_lr: Learning rate for phase (height in meters)
        phase_iters: Number of iterations for phase optimization
        loss: Loss function configuration
        fab_enabled: Whether to run fabrication optimization
        fab_method: Algorithm for fab optimization
        fab_lr: Learning rate for fab (dose 0-255)
        fab_iters: Number of iterations for fab optimization
        optimizer_type: Underlying optimizer ('adam' or 'sgd')
        scheduler_enabled: Whether to use learning rate scheduler

    Note on learning rates:
        - Phase optimization: height in meters (~1e-6 scale), use lr ~1e-8 to 1e-9
        - Fab optimization: dose in 0-255 range, use lr ~100-200
    """
    # Phase optimization (Step 1)
    phase_method: OptMethod = OptMethod.SGD
    phase_lr: float = 1e-8
    phase_iters: int = 10000
    loss: LossConfig = field(default_factory=LossConfig)

    # Fabrication optimization (Step 2)
    fab_enabled: bool = False
    fab_method: OptMethod = OptMethod.SGD
    fab_lr: float = 200.0
    fab_iters: int = 25000

    # Advanced options
    optimizer_type: Literal['adam', 'sgd'] = 'adam'
    adam_eps: float = 1e-8
    scheduler_enabled: bool = False
    scheduler_step_size: int = 5000
    scheduler_gamma: float = 0.5

    # Global energy learning (for ignore-efficiency mode)
    global_energy_lr: Optional[float] = None

    # Pixel multiplier - groups DOE pixels to reduce effective resolution
    # This reduces maximum diffraction angle: θ_max = arcsin(λ / (2 × pixel_size × multiplier))
    # After optimization, the phase is expanded back to original resolution
    pixel_multiplier: int = 1

    # Simulation upsample factor - increases resolution during propagation
    simulation_upsample: int = 1

    def __post_init__(self):
        """Validate configuration."""
        if self.phase_lr <= 0:
            raise ValueError("phase_lr must be positive")
        if self.phase_iters <= 0:
            raise ValueError("phase_iters must be positive")
        if self.fab_enabled:
            if self.fab_lr <= 0:
                raise ValueError("fab_lr must be positive")
            if self.fab_iters <= 0:
                raise ValueError("fab_iters must be positive")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            'phase_method': self.phase_method.value,
            'phase_lr': self.phase_lr,
            'phase_iters': self.phase_iters,
            'loss': self.loss.to_dict(),
            'fab_enabled': self.fab_enabled,
            'optimizer_type': self.optimizer_type,
            'pixel_multiplier': self.pixel_multiplier,
            'simulation_upsample': self.simulation_upsample,
        }
        if self.fab_enabled:
            result['fab_method'] = self.fab_method.value
            result['fab_lr'] = self.fab_lr
            result['fab_iters'] = self.fab_iters
        if self.global_energy_lr:
            result['global_energy_lr'] = self.global_energy_lr
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OptimizationConfig':
        """Create from dictionary."""
        loss_data = data.get('loss', {})
        loss_type = LossType(loss_data.get('loss_type', 'L2'))
        loss = LossConfig(
            loss_type=loss_type,
            weights=loss_data.get('weights'),
            roi_enabled=loss_data.get('roi_enabled', False),
            focal_params=loss_data.get('focal_params'),
        )

        return cls(
            phase_method=OptMethod(data.get('phase_method', 'SGD')),
            phase_lr=data.get('phase_lr', 1e-8),
            phase_iters=data.get('phase_iters', 10000),
            loss=loss,
            fab_enabled=data.get('fab_enabled', False),
            fab_method=OptMethod(data.get('fab_method', 'SGD')),
            fab_lr=data.get('fab_lr', 200.0),
            fab_iters=data.get('fab_iters', 25000),
            optimizer_type=data.get('optimizer_type', 'adam'),
            global_energy_lr=data.get('global_energy_lr'),
            pixel_multiplier=data.get('pixel_multiplier', 1),
            simulation_upsample=data.get('simulation_upsample', 1),
        )

    @classmethod
    def default_for_splitter(cls) -> 'OptimizationConfig':
        """Default configuration for splitter DOEs."""
        return cls(
            phase_method=OptMethod.SGD,
            phase_lr=1e-8,
            phase_iters=10000,
            loss=LossConfig(loss_type=LossType.L2),
        )

    @classmethod
    def default_for_lens(cls) -> 'OptimizationConfig':
        """Default configuration for lens DOEs."""
        return cls(
            phase_method=OptMethod.SGD,
            phase_lr=1e-8,
            phase_iters=15000,
            loss=LossConfig(loss_type=LossType.FOCAL_EFFICIENCY),
        )

    @classmethod
    def default_for_diffuser(cls) -> 'OptimizationConfig':
        """Default configuration for diffuser DOEs."""
        return cls(
            phase_method=OptMethod.GS,
            phase_lr=1e-8,
            phase_iters=5000,
            loss=LossConfig(loss_type=LossType.L2),
        )
