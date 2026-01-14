"""
Base wizard class and output structure.

Wizards convert high-level user parameters to structured parameters
suitable for optimization. The wizard layer provides:
- Parameter validation and constraint checking
- Automatic computation of derived values (period, tolerance limit, etc.)
- Target pattern generation
- Propagator configuration
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union, TYPE_CHECKING
import torch
import numpy as np

from ..params.base import StructuredParams, PropagatorConfig
from ..params.optimization import OptimizationConfig
from ..api.errors import ValidationWarning

if TYPE_CHECKING:
    from ..api.schemas import UserInputSchema


@dataclass
class WizardOutput:
    """Output from wizard parameter generation.

    Contains all computed parameters and configurations needed
    for optimization.

    Attributes:
        structured_params: Structured parameters (FFTParams, SFRParams, or ASMParams)
        propagator_config: Configuration for propagator creation
        optimization_config: Optimization algorithm configuration
        target_pattern: Target amplitude pattern [1, C, H, W]
        computed_values: Dictionary of computed values for frontend display
            - period: Period in meters (for splitters)
            - period_pixels: Period in pixels
            - tolerance_limit: Computed tolerance limit
            - max_pixel_multiplier: Maximum pixel multiplier option
            - working_orders: List of working orders (for splitters)
            - strategy: Finite distance strategy name
        warnings: List of validation warnings
        metadata: Additional metadata for the specific DOE type
    """
    structured_params: StructuredParams
    propagator_config: PropagatorConfig
    optimization_config: OptimizationConfig
    target_pattern: torch.Tensor
    computed_values: Dict[str, Any] = field(default_factory=dict)
    warnings: List[ValidationWarning] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'structured_params': self.structured_params.to_dict(),
            'propagator_config': {
                'prop_type': self.propagator_config.prop_type.value,
                'feature_size': self.propagator_config.feature_size,
                'num_channels': self.propagator_config.num_channels,
            },
            'optimization_config': self.optimization_config.to_dict(),
            'target_pattern_shape': list(self.target_pattern.shape),
            'computed_values': self.computed_values,
            'warnings': [w.to_dict() for w in self.warnings],
            'metadata': self.metadata,
        }


class BaseWizard(ABC):
    """Base class for DOE-type-specific wizards.

    Each wizard handles:
    1. Parameter extraction from user input
    2. Physical constraint computation
    3. Structured parameter creation
    4. Target pattern generation
    5. Propagator and optimizer configuration

    Subclasses must implement:
    - generate_params(): Main parameter generation
    - get_constraints(): Return constraints for frontend
    """

    def __init__(self, max_resolution: int = 2000):
        """Initialize wizard.

        Args:
            max_resolution: Maximum allowed simulation resolution
        """
        self.max_resolution = max_resolution
        self.device = self._get_device()
        self.dtype = torch.float32

    def _get_device(self) -> torch.device:
        """Get torch device with robust CUDA error handling."""
        import os
        import gc

        # Check if CUDA is explicitly disabled via environment variable
        if os.environ.get('CUDA_VISIBLE_DEVICES', '') == '' or os.environ.get('FORCE_CPU', ''):
            return torch.device('cpu')

        # Check CUDA availability first (this shouldn't raise)
        try:
            cuda_available = torch.cuda.is_available()
        except Exception:
            return torch.device('cpu')

        if not cuda_available:
            return torch.device('cpu')

        # Try to use CUDA with comprehensive error handling
        try:
            gc.collect()

            # Each CUDA operation wrapped individually
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            try:
                torch.cuda.init()
            except Exception:
                pass

            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

            try:
                torch.cuda.synchronize()
            except Exception:
                # Sync failed - CUDA is likely broken
                return torch.device('cpu')

            # The actual test - create tensor and compute
            test = torch.zeros(1, device='cuda')
            _ = test + 1
            torch.cuda.synchronize()
            del test

            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            return torch.device('cuda')

        except Exception:
            # Any CUDA error - fall back to CPU silently
            try:
                gc.collect()
            except Exception:
                pass
            return torch.device('cpu')

    @abstractmethod
    def generate_params(
        self,
        user_input: Dict[str, Any],
        device: torch.device = None
    ) -> WizardOutput:
        """Generate structured parameters from user input.

        Args:
            user_input: User-provided parameters (from frontend JSON)
            device: Torch device for pattern generation

        Returns:
            WizardOutput with all computed parameters

        Raises:
            ValueError: If required parameters are missing or invalid
        """
        pass

    @abstractmethod
    def get_constraints(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Get constraints and limits for frontend validation.

        Returns dict with:
        - min/max values for numeric fields
        - allowed values for enums
        - computed limits based on physics

        Args:
            user_input: Partial user input for context-dependent constraints

        Returns:
            Dictionary of constraints
        """
        pass

    def _create_optimization_config(
        self,
        user_input: Dict[str, Any]
    ) -> OptimizationConfig:
        """Create optimization config from user input.

        Args:
            user_input: User input with 'optimization' key

        Returns:
            OptimizationConfig instance
        """
        from ..params.optimization import OptMethod, LossType, LossConfig

        opt_input = user_input.get('optimization', {})

        # Loss config
        loss_type_str = opt_input.get('loss_type', 'L2')
        try:
            loss_type = LossType(loss_type_str)
        except ValueError:
            loss_type = LossType.L2

        loss_config = LossConfig(
            loss_type=loss_type,
            roi_enabled=opt_input.get('roi_enabled', False),
        )

        # Optimization config
        phase_method_str = opt_input.get('phase_method', 'SGD')
        try:
            phase_method = OptMethod(phase_method_str)
        except ValueError:
            phase_method = OptMethod.SGD

        return OptimizationConfig(
            phase_method=phase_method,
            phase_lr=opt_input.get('phase_lr', 3e-9),
            phase_iters=opt_input.get('phase_iters', 1000),
            loss=loss_config,
            fab_enabled=opt_input.get('fab_enabled', False),
            fab_lr=opt_input.get('fab_lr', 200.0),
            fab_iters=opt_input.get('fab_iters', 25000),
            pixel_multiplier=opt_input.get('pixel_multiplier', 1),
            simulation_upsample=opt_input.get('simulation_upsample', 1),
        )

    def _compute_max_pixel_multiplier(
        self,
        base_pixels: int,
        device_pixels: int
    ) -> int:
        """Compute maximum allowed pixel multiplier.

        Args:
            base_pixels: Base resolution (period or DOE size)
            device_pixels: Total device pixels

        Returns:
            Maximum multiplier that keeps simulation within limits
        """
        # Multiplier should:
        # 1. Keep device pixels reasonable
        # 2. Keep base pixels within max_resolution
        max_from_base = self.max_resolution // base_pixels
        max_from_device = self.max_resolution // (device_pixels // base_pixels)

        return max(1, min(max_from_base, 10))  # Cap at 10

    def _compute_tolerance_limit(
        self,
        wavelength: float,
        period: float,
        target_span: float
    ) -> float:
        """Compute the physical tolerance limit.

        The tolerance limit is the minimum achievable tolerance based on
        the k-space sampling imposed by the period.

        Args:
            wavelength: Wavelength in meters
            period: Period in meters
            target_span: Target angular span in radians

        Returns:
            Minimum tolerance (0-1)
        """
        if period <= 0 or target_span <= 0:
            return 0.0

        # K-space step: delta_theta = wavelength / period
        k_step = wavelength / period

        # Tolerance = k_step / (2 * target_span)
        tolerance_limit = k_step / (2 * target_span)

        return min(1.0, tolerance_limit)

    def _validate_sampling(
        self,
        wavelength: float,
        pixel_size: float,
        warnings: List[ValidationWarning]
    ) -> None:
        """Check Nyquist sampling condition.

        Args:
            wavelength: Wavelength in meters
            pixel_size: Pixel size in meters
            warnings: List to append warnings to
        """
        from ..api.errors import WarningCode

        # Maximum angle without aliasing: sin(theta) < lambda / (2*p)
        max_sin = wavelength / (2 * pixel_size)

        if max_sin < 1:
            max_angle = np.arcsin(max_sin)
            if max_angle < np.pi / 6:  # Less than 30 degrees
                warnings.append(ValidationWarning(
                    code=WarningCode.ALIASING_RISK,
                    message=f"Maximum diffraction angle limited to {np.degrees(max_angle):.1f}°",
                    suggestion="Use smaller pixel size or shorter wavelength"
                ))
