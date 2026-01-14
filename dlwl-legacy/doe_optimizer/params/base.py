"""
Base classes for structured parameters.

This module defines the foundation for the new parameter system:
- PropagationType: Enum for propagation algorithm selection
- PhysicalConstants: Immutable physical parameters
- PropagatorConfig: Configuration for building propagators
- StructuredParams: Type alias for all parameter types
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union, TYPE_CHECKING
from enum import Enum
import numpy as np

if TYPE_CHECKING:
    from .fft_params import FFTParams
    from .sfr_params import SFRParams
    from .asm_params import ASMParams


class PropagationType(Enum):
    """Propagation algorithm type.

    Determines which physical model is used for light propagation:
    - FFT: Far-field / k-space propagation (infinite distance)
    - SFR: Scaled Fresnel propagation (large target, finite distance)
    - ASM: Angular Spectrum Method (small target, finite distance)
    """
    FFT = "fft"  # Type A: k-space / infinite distance
    SFR = "sfr"  # Type B: large target face finite distance
    ASM = "asm"  # Type C: small target face finite distance

    @classmethod
    def from_string(cls, value: str) -> 'PropagationType':
        """Create from string value."""
        mapping = {
            'fft': cls.FFT,
            'FFT': cls.FFT,
            'sfr': cls.SFR,
            'SFR': cls.SFR,
            'asm': cls.ASM,
            'ASM': cls.ASM,
        }
        if value not in mapping:
            raise ValueError(f"Unknown propagation type: {value}")
        return mapping[value]


@dataclass(frozen=True)
class PhysicalConstants:
    """Physical constants for DOE design (immutable).

    All values use SI units (meters, radians).

    Attributes:
        wavelength: Working wavelength in meters
        refraction_index: Refractive index of DOE material
        pixel_size: Fabrication pixel size in meters (fixed by equipment)
    """
    wavelength: float
    refraction_index: float = 1.62
    pixel_size: float = 0.5e-6

    def __post_init__(self):
        """Validate physical parameters."""
        if self.wavelength <= 0:
            raise ValueError("wavelength must be positive")
        if self.refraction_index <= 1:
            raise ValueError("refraction_index must be > 1")
        if self.pixel_size <= 0:
            raise ValueError("pixel_size must be positive")

    @property
    def height_2pi(self) -> float:
        """Height corresponding to 2*pi phase shift in meters."""
        return self.wavelength / (self.refraction_index - 1)

    def height_to_phase(self, height: float) -> float:
        """Convert height to phase (radians)."""
        return 2 * np.pi * height / self.height_2pi

    def phase_to_height(self, phase: float) -> float:
        """Convert phase to height (meters)."""
        return phase * self.height_2pi / (2 * np.pi)


@dataclass
class PropagatorConfig:
    """Configuration for propagator creation.

    This is passed to PropagatorBuilder to create the appropriate
    propagation function based on the parameter type.

    Attributes:
        prop_type: Type of propagation algorithm
        feature_size: (dy, dx) pixel size in meters
        wavelength: Wavelength(s) in meters
        working_distance: Propagation distance(s) (None for FFT)
        output_size: Physical output size in meters (for SFR)
        output_resolution: Output resolution in pixels
        num_channels: Number of wavelength/distance channels
        precompute_kernels: Whether to precompute propagation kernels
    """
    prop_type: PropagationType
    feature_size: Tuple[float, float]
    wavelength: Union[float, np.ndarray]
    working_distance: Optional[Union[float, np.ndarray]] = None
    output_size: Optional[Tuple[float, float]] = None
    output_resolution: Optional[Tuple[int, int]] = None
    num_channels: int = 1
    precompute_kernels: bool = True

    def __post_init__(self):
        """Validate and normalize parameters."""
        # Normalize wavelength to array
        if isinstance(self.wavelength, (int, float)):
            self._wavelength_arr = np.array([self.wavelength])
        else:
            self._wavelength_arr = np.atleast_1d(np.array(self.wavelength))

        # Normalize working_distance
        if self.working_distance is None:
            self._working_distance_arr = None
        elif isinstance(self.working_distance, (int, float)):
            self._working_distance_arr = np.array([self.working_distance])
        else:
            self._working_distance_arr = np.atleast_1d(np.array(self.working_distance))

        # Update num_channels
        if self._working_distance_arr is not None:
            self.num_channels = max(
                len(self._wavelength_arr),
                len(self._working_distance_arr)
            )
        else:
            self.num_channels = len(self._wavelength_arr)

    @property
    def wavelength_array(self) -> np.ndarray:
        """Get wavelength as [1, C, 1, 1] array for broadcasting."""
        arr = self._wavelength_arr
        if len(arr) < self.num_channels:
            arr = np.tile(arr, (self.num_channels // len(arr) + 1))[:self.num_channels]
        return arr.reshape(1, -1, 1, 1)

    @property
    def working_distance_array(self) -> Optional[np.ndarray]:
        """Get working distance as [1, C, 1, 1] array for broadcasting."""
        if self._working_distance_arr is None:
            return None
        arr = self._working_distance_arr
        if len(arr) < self.num_channels:
            arr = np.tile(arr, (self.num_channels // len(arr) + 1))[:self.num_channels]
        return arr.reshape(1, -1, 1, 1)


# Type alias for all structured parameter types
StructuredParams = Union['FFTParams', 'SFRParams', 'ASMParams']


def get_prop_type(params: StructuredParams) -> PropagationType:
    """Get propagation type from structured parameters.

    Args:
        params: Any structured parameter type

    Returns:
        PropagationType for the parameter
    """
    from .fft_params import FFTParams
    from .sfr_params import SFRParams
    from .asm_params import ASMParams

    if isinstance(params, FFTParams):
        return PropagationType.FFT
    elif isinstance(params, SFRParams):
        return PropagationType.SFR
    elif isinstance(params, ASMParams):
        return PropagationType.ASM
    else:
        raise TypeError(f"Unknown parameter type: {type(params)}")
