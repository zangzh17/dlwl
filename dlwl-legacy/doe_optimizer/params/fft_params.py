"""
Type A: k-space / infinite distance parameters.

FFTParams is used for:
- Infinite-distance splitters (both natural and uniform grid)
- Infinite-distance diffusers
- Strategy 2 finite-distance DOEs (periodic + Fresnel overlay)

The key characteristic is that optimization happens on a single period,
and the k-space (angle space) is uniformly sampled.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import torch

from .base import PhysicalConstants, PropagatorConfig, PropagationType


@dataclass
class FFTParams:
    """Type A: k-space / infinite distance parameters.

    For periodic DOEs where optimization is done on a single period unit.
    The output is in k-space (angular frequency domain).

    Attributes:
        period_pixels: Optimization unit size (H, W) in pixels
        doe_total_pixels: Full device size (H, W) in pixels
        physical: Physical constants (wavelength, refractive index, pixel size)
        upsample_factor: Upsampling factor for simulation
        aperture_type: Device aperture shape ('square' or 'circular')

    Example:
        # 5x5 splitter with natural grid
        params = FFTParams(
            period_pixels=(17, 17),  # Computed by wizard
            doe_total_pixels=(2000, 2000),
            physical=PhysicalConstants(
                wavelength=532e-9,
                refraction_index=1.62,
                pixel_size=0.5e-6
            ),
            upsample_factor=2
        )
    """
    period_pixels: Tuple[int, int]
    doe_total_pixels: Tuple[int, int]
    physical: PhysicalConstants
    upsample_factor: int = 1
    aperture_type: str = 'square'

    # Computed at init
    _target_pattern: Optional[torch.Tensor] = field(default=None, repr=False)

    def __post_init__(self):
        """Validate parameters."""
        h, w = self.period_pixels
        if h <= 0 or w <= 0:
            raise ValueError("period_pixels must be positive")

        dh, dw = self.doe_total_pixels
        if dh <= 0 or dw <= 0:
            raise ValueError("doe_total_pixels must be positive")

        if self.upsample_factor < 1:
            raise ValueError("upsample_factor must be >= 1")

        if self.aperture_type not in ('square', 'circular'):
            raise ValueError("aperture_type must be 'square' or 'circular'")

    @property
    def period_size(self) -> Tuple[float, float]:
        """Physical period size in meters."""
        h, w = self.period_pixels
        ps = self.physical.pixel_size
        return (h * ps, w * ps)

    @property
    def feature_size(self) -> Tuple[float, float]:
        """Feature size (pixel size * upsample) for propagation."""
        ps = self.physical.pixel_size / self.upsample_factor
        return (ps, ps)

    @property
    def simulation_pixels(self) -> Tuple[int, int]:
        """Resolution used during simulation (with upsampling)."""
        h, w = self.period_pixels
        return (h * self.upsample_factor, w * self.upsample_factor)

    @property
    def num_periods(self) -> Tuple[int, int]:
        """Number of periods that fit in the full device."""
        dh, dw = self.doe_total_pixels
        ph, pw = self.period_pixels
        return (dh // ph, dw // pw)

    @property
    def k_spacing(self) -> Tuple[float, float]:
        """K-space (angular) spacing in radians.

        The angular resolution in k-space is determined by the period size:
        delta_theta = wavelength / period_size
        """
        wl = self.physical.wavelength
        ph, pw = self.period_size
        return (wl / ph, wl / pw)

    def set_target_pattern(self, pattern: torch.Tensor) -> None:
        """Set the target pattern for optimization.

        Args:
            pattern: Target amplitude pattern [1, C, H, W]
                     H, W should match period_pixels (or simulation_pixels if upsampled)
        """
        self._target_pattern = pattern

    @property
    def target_pattern(self) -> Optional[torch.Tensor]:
        """Get the target pattern."""
        return self._target_pattern

    def to_propagator_config(self) -> PropagatorConfig:
        """Create PropagatorConfig for this parameter set."""
        return PropagatorConfig(
            prop_type=PropagationType.FFT,
            feature_size=self.feature_size,
            wavelength=self.physical.wavelength,
            working_distance=None,
            output_resolution=self.simulation_pixels,
            num_channels=1,
            precompute_kernels=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'type': 'FFTParams',
            'period_pixels': self.period_pixels,
            'doe_total_pixels': self.doe_total_pixels,
            'wavelength': self.physical.wavelength,
            'refraction_index': self.physical.refraction_index,
            'pixel_size': self.physical.pixel_size,
            'upsample_factor': self.upsample_factor,
            'aperture_type': self.aperture_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FFTParams':
        """Create from dictionary."""
        physical = PhysicalConstants(
            wavelength=data['wavelength'],
            refraction_index=data.get('refraction_index', 1.62),
            pixel_size=data.get('pixel_size', 0.5e-6),
        )
        return cls(
            period_pixels=tuple(data['period_pixels']),
            doe_total_pixels=tuple(data['doe_total_pixels']),
            physical=physical,
            upsample_factor=data.get('upsample_factor', 1),
            aperture_type=data.get('aperture_type', 'square'),
        )
