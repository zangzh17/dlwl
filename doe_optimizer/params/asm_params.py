"""
Type C: Small target face finite distance parameters.

ASMParams is used for:
- Strategy 1 finite-distance splitters (target size ~ DOE size)
- Near-field propagation where output size is similar to input
- Non-periodic DOE optimization

The Angular Spectrum Method (ASM) provides exact diffraction calculation
for finite distance propagation with configurable output size.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
import torch

from .base import PhysicalConstants, PropagatorConfig, PropagationType


@dataclass
class ASMParams:
    """Type C: Finite distance parameters with configurable output size.

    For DOEs with finite propagation distance. Similar interface to SFRParams
    with output_size and output_resolution for flexibility.

    Attributes:
        doe_pixels: DOE size (H, W) in pixels
        physical: Physical constants
        working_distances: Propagation distance(s) in meters
        target_size: Physical target size (H, W) in meters. If None, uses DOE size.
        target_resolution: Target resolution (H, W) in pixels. If None, derived from target_size.
        upsample_factor: Upsampling factor for simulation
        aperture_type: Device aperture shape
        use_linear_conv: Use linear convolution (default True for accuracy)

    Example:
        # Finite-distance splitter with 1mm target on 256um DOE
        params = ASMParams(
            doe_pixels=(256, 256),
            physical=PhysicalConstants(wavelength=532e-9, pixel_size=1e-6),
            working_distances=[0.01],  # 10mm
            target_size=(0.001, 0.001),  # 1mm x 1mm target
            target_resolution=(256, 256),  # 256x256 output pixels
        )
    """
    doe_pixels: Tuple[int, int]
    physical: PhysicalConstants
    working_distances: List[float]
    target_size: Optional[Tuple[float, float]] = None  # Physical size in meters
    target_resolution: Optional[Tuple[int, int]] = None  # Output pixel count
    upsample_factor: int = 1
    aperture_type: str = 'square'
    use_linear_conv: bool = True

    # Legacy compatibility: target_pixels (will be computed from target_size)
    target_pixels: Optional[Tuple[int, int]] = field(default=None, repr=False)

    # Target pattern (set by wizard)
    _target_pattern: Optional[torch.Tensor] = field(default=None, repr=False)

    def __post_init__(self):
        """Validate parameters."""
        h, w = self.doe_pixels
        if h <= 0 or w <= 0:
            raise ValueError("doe_pixels must be positive")

        if not self.working_distances or any(d <= 0 for d in self.working_distances):
            raise ValueError("working_distances must be positive")

        # If target_size not specified, default to DOE size
        if self.target_size is None:
            self.target_size = self.doe_size

        # If target_resolution not specified, derive from target_size
        if self.target_resolution is None:
            ps = self.physical.pixel_size
            self.target_resolution = (
                int(np.ceil(self.target_size[0] / ps)),
                int(np.ceil(self.target_size[1] / ps))
            )

        # Compute target_pixels for legacy compatibility (internal simulation pixels)
        # This is the pixel count at DOE's pixel_size for the simulation area
        ps = self.physical.pixel_size
        sim_h = max(int(np.ceil(self.target_size[0] / ps)), h)
        sim_w = max(int(np.ceil(self.target_size[1] / ps)), w)
        self.target_pixels = (sim_h, sim_w)

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
    def output_pixels(self) -> Tuple[int, int]:
        """Output resolution (target_resolution with upsampling)."""
        h, w = self.target_resolution
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
                     H, W should match target_resolution * upsample_factor
        """
        self._target_pattern = pattern

    @property
    def target_pattern(self) -> Optional[torch.Tensor]:
        """Get the target pattern."""
        return self._target_pattern

    def to_propagator_config(self) -> PropagatorConfig:
        """Create PropagatorConfig for this parameter set.

        Includes output_size for ASM to support arbitrary target sizes.
        """
        # Apply upsample_factor to target_size
        target_size_upsampled = (
            self.target_size[0],  # Physical size doesn't change with upsample
            self.target_size[1]
        )
        return PropagatorConfig(
            prop_type=PropagationType.ASM,
            feature_size=self.feature_size,
            wavelength=self.physical.wavelength,
            working_distance=self._working_distances_arr,
            output_size=target_size_upsampled,  # NEW: physical output size
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
            'target_size': self.target_size,  # NEW: physical size
            'target_resolution': self.target_resolution,  # NEW: output resolution
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
        # Support both old (target_pixels) and new (target_size/target_resolution) format
        target_size = data.get('target_size')
        target_resolution = data.get('target_resolution')

        # Legacy: if target_pixels provided but not target_size, compute from pixels
        if target_size is None and 'target_pixels' in data:
            tp = data['target_pixels']
            target_size = (tp[0] * physical.pixel_size, tp[1] * physical.pixel_size)
            target_resolution = tuple(tp)

        return cls(
            doe_pixels=tuple(data['doe_pixels']),
            physical=physical,
            working_distances=data['working_distances'],
            target_size=tuple(target_size) if target_size else None,
            target_resolution=tuple(target_resolution) if target_resolution else None,
            upsample_factor=data.get('upsample_factor', 1),
            aperture_type=data.get('aperture_type', 'square'),
            use_linear_conv=data.get('use_linear_conv', True),
        )
