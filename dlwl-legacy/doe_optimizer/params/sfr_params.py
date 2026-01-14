"""
Type B: Large target face finite distance parameters.

SFRParams is used for:
- Finite-distance diffusers where target is much larger than DOE
- Large-area projectors
- Any DOE where output size differs significantly from input size

The Scaled Fresnel (SFR) algorithm allows arbitrary output size/resolution
through zoom-FFT techniques.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any, Union
import numpy as np
import torch

from .base import PhysicalConstants, PropagatorConfig, PropagationType


@dataclass
class SFRParams:
    """Type B: Large target face finite distance parameters.

    For DOEs where the target plane is larger than the DOE itself,
    requiring zoom-FFT techniques (Scaled Fresnel).

    Attributes:
        doe_pixels: DOE size (H, W) in pixels
        physical: Physical constants
        working_distances: Propagation distance(s) in meters (can be list)
        target_size: Physical size of target plane (H, W) in meters
        target_resolution: Resolution of target plane (H, W) in pixels
        upsample_factor: Upsampling factor for DOE simulation
        aperture_type: Device aperture shape

    Example:
        # Finite-distance diffuser
        params = SFRParams(
            doe_pixels=(1000, 1000),
            physical=PhysicalConstants(wavelength=532e-9),
            working_distances=[0.1],  # 10cm
            target_size=(0.05, 0.05),  # 5cm x 5cm target
            target_resolution=(512, 512),
            upsample_factor=2
        )
    """
    doe_pixels: Tuple[int, int]
    physical: PhysicalConstants
    working_distances: List[float]
    target_size: Tuple[float, float]
    target_resolution: Tuple[int, int]
    upsample_factor: int = 1
    aperture_type: str = 'square'

    # Target pattern (set by wizard)
    _target_pattern: Optional[torch.Tensor] = field(default=None, repr=False)

    def __post_init__(self):
        """Validate parameters."""
        h, w = self.doe_pixels
        if h <= 0 or w <= 0:
            raise ValueError("doe_pixels must be positive")

        if not self.working_distances or any(d <= 0 for d in self.working_distances):
            raise ValueError("working_distances must be positive")

        th, tw = self.target_size
        if th <= 0 or tw <= 0:
            raise ValueError("target_size must be positive")

        rh, rw = self.target_resolution
        if rh <= 0 or rw <= 0:
            raise ValueError("target_resolution must be positive")

        # Convert to numpy array
        self._working_distances_arr = np.array(self.working_distances)

    @property
    def doe_size(self) -> Tuple[float, float]:
        """Physical DOE size in meters."""
        h, w = self.doe_pixels
        ps = self.physical.pixel_size
        return (h * ps, w * ps)

    @property
    def feature_size(self) -> Tuple[float, float]:
        """Feature size for propagation."""
        ps = self.physical.pixel_size / self.upsample_factor
        return (ps, ps)

    @property
    def simulation_pixels(self) -> Tuple[int, int]:
        """DOE resolution during simulation (with upsampling)."""
        h, w = self.doe_pixels
        return (h * self.upsample_factor, w * self.upsample_factor)

    @property
    def target_pixel_size(self) -> Tuple[float, float]:
        """Pixel size in the target plane."""
        th, tw = self.target_size
        rh, rw = self.target_resolution
        return (th / rh, tw / rw)

    @property
    def num_channels(self) -> int:
        """Number of wavelength/distance channels."""
        return len(self.working_distances)

    @property
    def zoom_factors(self) -> Tuple[float, float]:
        """Zoom factors for SFR (target_size / doe_size)."""
        dh, dw = self.doe_size
        th, tw = self.target_size
        return (th / dh, tw / dw)

    def set_target_pattern(self, pattern: torch.Tensor) -> None:
        """Set the target pattern.

        Args:
            pattern: Target amplitude [1, C, H, W]
                     C = num_channels (for multi-distance optimization)
                     H, W should match target_resolution
        """
        self._target_pattern = pattern

    @property
    def target_pattern(self) -> Optional[torch.Tensor]:
        """Get the target pattern."""
        return self._target_pattern

    def to_propagator_config(self) -> PropagatorConfig:
        """Create PropagatorConfig for this parameter set."""
        return PropagatorConfig(
            prop_type=PropagationType.SFR,
            feature_size=self.feature_size,
            wavelength=self.physical.wavelength,
            working_distance=self._working_distances_arr,
            output_size=self.target_size,
            output_resolution=self.target_resolution,
            num_channels=self.num_channels,
            precompute_kernels=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'type': 'SFRParams',
            'doe_pixels': self.doe_pixels,
            'wavelength': self.physical.wavelength,
            'refraction_index': self.physical.refraction_index,
            'pixel_size': self.physical.pixel_size,
            'working_distances': self.working_distances,
            'target_size': self.target_size,
            'target_resolution': self.target_resolution,
            'upsample_factor': self.upsample_factor,
            'aperture_type': self.aperture_type,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SFRParams':
        """Create from dictionary."""
        physical = PhysicalConstants(
            wavelength=data['wavelength'],
            refraction_index=data.get('refraction_index', 1.62),
            pixel_size=data.get('pixel_size', 0.5e-6),
        )
        return cls(
            doe_pixels=tuple(data['doe_pixels']),
            physical=physical,
            working_distances=data['working_distances'],
            target_size=tuple(data['target_size']),
            target_resolution=tuple(data['target_resolution']),
            upsample_factor=data.get('upsample_factor', 1),
            aperture_type=data.get('aperture_type', 'square'),
        )
