"""
Type C: Small target face finite distance parameters.

ASMParams is used for:
- Strategy 1 finite-distance splitters (target size ~ DOE size)
- Near-field propagation where output size is similar to input
- Non-periodic DOE optimization

The Angular Spectrum Method (ASM) provides exact diffraction calculation
when the output and input planes have similar physical sizes.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import torch

from .base import PhysicalConstants, PropagatorConfig, PropagationType


@dataclass
class ASMParams:
    """Type C: Small target face finite distance parameters.

    For DOEs where the target plane is similar in size to the DOE,
    suitable for direct ASM propagation.

    Attributes:
        doe_pixels: DOE size (H, W) in pixels
        physical: Physical constants
        working_distances: Propagation distance(s) in meters
        target_pixels: Target plane size (H, W) in pixels
                      Larger than doe_pixels uses zero-padding
        upsample_factor: Upsampling factor for simulation
        aperture_type: Device aperture shape
        use_linear_conv: Use linear convolution (default True for accuracy)

    Example:
        # Strategy 1 finite-distance splitter
        params = ASMParams(
            doe_pixels=(1000, 1000),
            physical=PhysicalConstants(wavelength=532e-9),
            working_distances=[0.05],  # 5cm
            target_pixels=(1200, 1200),  # Slight padding
            upsample_factor=2
        )
    """
    doe_pixels: Tuple[int, int]
    physical: PhysicalConstants
    working_distances: List[float]
    target_pixels: Tuple[int, int]
    upsample_factor: int = 1
    aperture_type: str = 'square'
    use_linear_conv: bool = True

    # Target pattern (set by wizard)
    _target_pattern: Optional[torch.Tensor] = field(default=None, repr=False)

    def __post_init__(self):
        """Validate parameters."""
        h, w = self.doe_pixels
        if h <= 0 or w <= 0:
            raise ValueError("doe_pixels must be positive")

        if not self.working_distances or any(d <= 0 for d in self.working_distances):
            raise ValueError("working_distances must be positive")

        th, tw = self.target_pixels
        if th <= 0 or tw <= 0:
            raise ValueError("target_pixels must be positive")

        # Target should be at least as large as DOE for linear convolution
        if self.use_linear_conv and (th < h or tw < w):
            raise ValueError("target_pixels must be >= doe_pixels for linear convolution")

        # Convert to numpy array
        self._working_distances_arr = np.array(self.working_distances)

    @property
    def doe_size(self) -> Tuple[float, float]:
        """Physical DOE size in meters."""
        h, w = self.doe_pixels
        ps = self.physical.pixel_size
        return (h * ps, w * ps)

    @property
    def target_size(self) -> Tuple[float, float]:
        """Physical target size in meters."""
        h, w = self.target_pixels
        ps = self.physical.pixel_size  # Same pixel size for ASM
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
    def output_pixels(self) -> Tuple[int, int]:
        """Output resolution (target pixels with upsampling)."""
        h, w = self.target_pixels
        return (h * self.upsample_factor, w * self.upsample_factor)

    @property
    def num_channels(self) -> int:
        """Number of wavelength/distance channels."""
        return len(self.working_distances)

    @property
    def padding_pixels(self) -> Tuple[int, int]:
        """Padding required to go from DOE to target size."""
        dh, dw = self.doe_pixels
        th, tw = self.target_pixels
        return ((th - dh) // 2, (tw - dw) // 2)

    def set_target_pattern(self, pattern: torch.Tensor) -> None:
        """Set the target pattern.

        Args:
            pattern: Target amplitude [1, C, H, W]
                     C = num_channels
                     H, W should match target_pixels * upsample_factor
        """
        self._target_pattern = pattern

    @property
    def target_pattern(self) -> Optional[torch.Tensor]:
        """Get the target pattern."""
        return self._target_pattern

    def to_propagator_config(self) -> PropagatorConfig:
        """Create PropagatorConfig for this parameter set."""
        return PropagatorConfig(
            prop_type=PropagationType.ASM,
            feature_size=self.feature_size,
            wavelength=self.physical.wavelength,
            working_distance=self._working_distances_arr,
            output_resolution=self.output_pixels,
            num_channels=self.num_channels,
            precompute_kernels=True,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'type': 'ASMParams',
            'doe_pixels': self.doe_pixels,
            'wavelength': self.physical.wavelength,
            'refraction_index': self.physical.refraction_index,
            'pixel_size': self.physical.pixel_size,
            'working_distances': self.working_distances,
            'target_pixels': self.target_pixels,
            'upsample_factor': self.upsample_factor,
            'aperture_type': self.aperture_type,
            'use_linear_conv': self.use_linear_conv,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ASMParams':
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
            target_pixels=tuple(data['target_pixels']),
            upsample_factor=data.get('upsample_factor', 1),
            aperture_type=data.get('aperture_type', 'square'),
            use_linear_conv=data.get('use_linear_conv', True),
        )
