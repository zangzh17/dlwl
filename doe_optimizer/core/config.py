"""
Configuration classes for DOE optimization.

This module defines all configuration dataclasses that replace the YAML-based configuration.
All parameters are explicitly typed and documented.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, List, Tuple, Union, Any
from enum import Enum
import numpy as np
import math


# ============================================================================
# Global Configuration
# ============================================================================

# Maximum resolution allowed for optimization/simulation (e.g., 2000x2000)
# This can be set externally to limit computational resources
MAX_OPTIMIZATION_RESOLUTION: int = 2000


class DOEType(Enum):
    """Supported DOE types."""
    SPLITTER_1D = "splitter_1d"          # 1D beam splitter
    SPLITTER_2D = "splitter_2d"          # 2D beam splitter (spot array)
    SPOT_PROJECTOR = "spot_projector"    # 2D spot projector with custom layout
    DIFFUSER = "diffuser"                # Homogenizer / diffuser
    LENS = "lens"                        # Diffractive lens
    LENS_ARRAY = "lens_array"            # Microlens array
    DEFLECTOR = "deflector"              # Beam deflector / blazed grating
    CUSTOM = "custom"                    # Custom pattern from image


class PropModel(Enum):
    """Propagation model types."""
    ASM = "ASM"    # Angular Spectrum Method - near field, input/output same size
    FFT = "FFT"    # Far field FFT - infinite distance, angle space
    SFR = "SFR"    # Scaled Fresnel - far field with adjustable output size


class SplitterMode(Enum):
    """Splitter/spot projector grid mode.

    Determines how spot positions are calculated relative to diffraction orders.
    """
    NATURAL_GRID = "natural"   # K-space uniform: natural diffraction orders
    UNIFORM_GRID = "uniform"   # Angle/size space uniform: snap to k-space grid


class FiniteDistanceStrategy(Enum):
    """Strategy for handling finite working distance in splitter/spot projector.

    Strategy 1 (ASM): Target size is similar to DOE size, use direct ASM propagation.
                      Non-periodic optimization, no tolerance parameter needed.

    Strategy 2 (PERIODIC_FRESNEL): Target size is much larger than DOE size.
                                   Treat as infinite distance + Fresnel phase overlay.
                                   Periodic optimization like infinite case, then add
                                   global focusing Fresnel phase to full device.
    """
    ASM = "asm"                        # Strategy 1: Direct ASM, small target
    PERIODIC_FRESNEL = "periodic_fresnel"  # Strategy 2: Periodic + Fresnel overlay


@dataclass
class PhysicalParams:
    """Physical parameters for DOE design.

    Attributes:
        wavelength: Working wavelength in meters (can be list for multi-wavelength)
        refraction_index: Refractive index of DOE material (can be list)
        working_distance: Propagation distance in meters. None means infinite (far field)
    """
    wavelength: Union[float, List[float]]
    refraction_index: Union[float, List[float]] = 1.62
    working_distance: Optional[Union[float, List[float]]] = None  # None = infinite

    def __post_init__(self):
        # Convert to numpy arrays for internal use
        self._wavelength_arr = np.atleast_1d(np.array(self.wavelength))
        self._refraction_index_arr = np.atleast_1d(np.array(self.refraction_index))
        if self.working_distance is not None:
            self._working_distance_arr = np.atleast_1d(np.array(self.working_distance))
        else:
            self._working_distance_arr = None

    @property
    def num_channels(self) -> int:
        """Number of wavelength/distance channels."""
        if self._working_distance_arr is not None:
            return max(len(self._wavelength_arr), len(self._working_distance_arr))
        return len(self._wavelength_arr)

    @property
    def wavelength_array(self) -> np.ndarray:
        """Get wavelength as numpy array with shape [1, C, 1, 1]."""
        arr = self._wavelength_arr
        if len(arr) < self.num_channels:
            arr = np.tile(arr, self.num_channels // len(arr) + 1)[:self.num_channels]
        return arr.reshape(1, -1, 1, 1)

    @property
    def refraction_index_array(self) -> np.ndarray:
        """Get refraction index as numpy array with shape [1, C, 1, 1]."""
        arr = self._refraction_index_arr
        if len(arr) < self.num_channels:
            arr = np.tile(arr, self.num_channels // len(arr) + 1)[:self.num_channels]
        return arr.reshape(1, -1, 1, 1)

    @property
    def working_distance_array(self) -> Optional[np.ndarray]:
        """Get working distance as numpy array with shape [1, C, 1, 1]."""
        if self._working_distance_arr is None:
            return None
        arr = self._working_distance_arr
        if len(arr) < self.num_channels:
            arr = np.tile(arr, self.num_channels // len(arr) + 1)[:self.num_channels]
        return arr.reshape(1, -1, 1, 1)

    def height_2pi(self, channel: int = 0) -> float:
        """Calculate height corresponding to 2pi phase shift."""
        wl = self._wavelength_arr[min(channel, len(self._wavelength_arr) - 1)]
        n = self._refraction_index_arr[min(channel, len(self._refraction_index_arr) - 1)]
        return float(wl / (n - 1))


@dataclass
class DeviceParams:
    """DOE device parameters.

    Attributes:
        diameter: Device diameter/size in meters
        shape: Device aperture shape
        pixel_size: Global fabrication pixel size in meters (fixed by equipment)
    """
    diameter: float
    shape: Literal["square", "circular"] = "square"
    pixel_size: float = 0.5e-6  # 0.5 um default

    @property
    def num_pixels(self) -> int:
        """Number of pixels across the device diameter."""
        return int(round(self.diameter / self.pixel_size))


@dataclass
class OptimizationParams:
    """Optimization algorithm parameters.

    Attributes:
        phase_method: Algorithm for phase optimization (step 1)
        phase_lr: Learning rate for phase optimization (default 1e-8 for height in meters)
        phase_iters: Number of iterations for phase optimization
        phase_pixel_multiplier: Pixel size multiplier for faster optimization
        fab_method: Algorithm for fabrication optimization (step 2)
        fab_lr: Learning rate for fabrication optimization (default 200 for dose 0-255)
        fab_iters: Number of iterations for fabrication optimization
        loss_type: Loss function type

    Note:
        Learning rates are tuned for the specific optimization variables:
        - Phase optimization: height in meters (um scale ~1e-6), use lr ~1e-8 to 1e-9
        - Fab optimization: dose in 0-255 range, use lr ~100-200
    """
    # Phase optimization (step 1)
    phase_method: Literal["SGD", "GS", "BS"] = "SGD"
    phase_lr: float = 1e-8  # Learning rate for height (meters)
    phase_iters: int = 10000
    phase_pixel_multiplier: int = 1  # User-selected multiplier

    # Fabrication optimization (step 2)
    fab_method: Literal["SGD", "GS", "BS"] = "SGD"
    fab_lr: float = 200.0  # Learning rate for dose (0-255)
    fab_iters: int = 25000

    # Loss function
    loss_type: Literal["L1", "L2"] = "L2"

    # Simulation upsampling
    simulation_upsample: int = 1  # Upsampling factor for phase during simulation/optimization
    eval_upsample: int = 1  # Upsampling factor for evaluation (can differ from optimization)

    # Advanced options
    optimizer_type: Literal["adam", "sgd"] = "adam"
    adam_eps: float = 1e-8


@dataclass
class FabricationCalibration:
    """Fabrication calibration data (GT/LP curves).

    This data is typically loaded from MATLAB .mat files and passed externally.

    Attributes:
        gt_data: Grayscale-to-thickness curve data (dict from .mat file)
        lp_data: Low-pass (MTF) curve data (dict from .mat file)
        lp_ratio: Scaling factor for LP curve
        dose_range: Valid dose range [min, max]
    """
    gt_data: dict
    lp_data: dict
    lp_ratio: float = 1.0
    dose_range: Tuple[float, float] = (0.0, 255.0)


@dataclass
class TargetParams:
    """Target pattern parameters.

    Different DOE types require different subsets of these parameters.
    """
    # Target specification type
    target_type: Literal["angle", "size"] = "angle"
    # Target span: (x, y) in radians for angle, meters for size
    target_span: Tuple[float, float] = (0.1, 0.1)

    # Tolerance (user-specified percentage, e.g., 0.05 = 5%)
    tolerance: float = 0.05

    # === Splitter / Spot Projector specific ===
    num_spots: Optional[Tuple[int, int]] = None  # (rows, cols) for 2D, (n, 1) for 1D
    spot_orders: Optional[List[int]] = None  # Specific diffraction orders to use
    splitter_mode: str = "natural"  # "natural" (k-space uniform) or "uniform" (angle space uniform)

    # === Diffuser specific ===
    diffuser_shape: Literal["square", "circular"] = "square"

    # === Lens specific ===
    focal_length: Optional[float] = None
    lens_type: Literal["normal", "cylindrical_x", "cylindrical_y"] = "normal"
    # Extended depth of focus: list of focal lengths
    focal_lengths: Optional[List[float]] = None

    # === Lens Array specific ===
    array_size: Optional[Tuple[int, int]] = None  # (rows, cols)

    # === Deflector specific ===
    deflection_angle: Optional[Tuple[float, float]] = None  # (theta_x, theta_y) in radians

    # === Custom pattern specific ===
    target_image: Optional[np.ndarray] = None  # Grayscale image array
    target_resolution: Optional[Tuple[int, int]] = None  # Resize resolution

    # === Output resolution ===
    roi_resolution: Optional[Tuple[int, int]] = None  # Loss calculation region
    output_size: Optional[Tuple[float, float]] = None  # Physical output size for SFR


@dataclass
class DOEConfig:
    """Complete DOE optimization configuration.

    This is the main configuration class that combines all parameters.
    """
    doe_type: DOEType
    physical: PhysicalParams
    device: DeviceParams
    target: TargetParams
    optimization: OptimizationParams = field(default_factory=OptimizationParams)
    fabrication: Optional[FabricationCalibration] = None
    enable_fab_optimization: bool = False

    # Computed properties (filled during validation)
    _prop_model: PropModel = field(default=PropModel.FFT, init=False)
    _slm_resolution: Tuple[int, int] = field(default=(1000, 1000), init=False)
    _tolerance_limit: float = field(default=0.0, init=False)
    _max_pixel_multiplier: int = field(default=1, init=False)

    def __post_init__(self):
        """Validate configuration and compute derived values."""
        self._validate()
        self._compute_derived_values()

    def _validate(self):
        """Validate configuration parameters."""
        # Check fabrication params when fab optimization is enabled
        if self.enable_fab_optimization and self.fabrication is None:
            raise ValueError("Fabrication calibration data required when enable_fab_optimization=True")

        # Validate pixel multiplier
        if self.optimization.phase_pixel_multiplier < 1:
            raise ValueError("phase_pixel_multiplier must be >= 1")

    def _compute_derived_values(self):
        """Compute derived values from configuration."""
        # Determine propagation model
        if self.physical.working_distance is None:
            self._prop_model = PropModel.FFT
        elif self.target.output_size is not None:
            self._prop_model = PropModel.SFR
        else:
            # For splitters, check finite distance strategy
            # Strategy 2 (PERIODIC_FRESNEL) should use FFT like infinite case
            if self.is_splitter():
                strategy = self.get_finite_distance_strategy()
                if strategy == FiniteDistanceStrategy.PERIODIC_FRESNEL:
                    self._prop_model = PropModel.FFT
                else:
                    self._prop_model = PropModel.ASM
            else:
                self._prop_model = PropModel.ASM

        # Compute SLM resolution
        base_pixels = self.device.num_pixels
        self._slm_resolution = (base_pixels, base_pixels)

        # Compute tolerance limit
        self._tolerance_limit = self._compute_tolerance_limit()

        # Compute max pixel multiplier
        self._max_pixel_multiplier = self._compute_max_pixel_multiplier()

        # Validate user's pixel multiplier
        if self.optimization.phase_pixel_multiplier > self._max_pixel_multiplier:
            raise ValueError(
                f"phase_pixel_multiplier ({self.optimization.phase_pixel_multiplier}) exceeds "
                f"maximum allowed ({self._max_pixel_multiplier}) for given diffraction angle"
            )

    def _compute_tolerance_limit(self) -> float:
        """Compute the physical tolerance limit based on device and target parameters.

        Returns minimum achievable tolerance percentage.
        """
        wavelength = self.physical._wavelength_arr.min()
        D = self.device.diameter

        if self.physical.working_distance is None:
            # Infinite distance: angle space
            # T_min = lambda / (2 * D * delta_sin_theta)
            span_angle = max(self.target.target_span)
            delta_sin_theta = 2 * math.sin(span_angle / 2)  # Full FOV in sin(theta)
            if delta_sin_theta > 0:
                return wavelength / (2 * D * delta_sin_theta)
            return 0.0
        else:
            # Finite distance: size space
            # T_min = lambda * z / (2 * D * S)
            z = np.min(self.physical._working_distance_arr)
            S = max(self.target.target_span)  # Physical size
            if S > 0:
                return wavelength * z / (2 * D * S)
            return 0.0

    def _compute_max_pixel_multiplier(self) -> int:
        """Compute maximum allowed pixel size multiplier.

        Based on Nyquist sampling theorem for maximum diffraction angle.
        Only applicable for FFT and SFR modes.
        """
        if self._prop_model == PropModel.ASM:
            # ASM mode doesn't support downsampling
            return 1

        wavelength = self.physical._wavelength_arr.min()
        p_global = self.device.pixel_size

        # Compute theta_max based on target
        if self.physical.working_distance is None:
            # Infinite: use target angle span
            theta_max = max(self.target.target_span) / 2
        else:
            # Finite: compute from target size and distance
            z = np.min(self.physical._working_distance_arr)
            S_target = max(self.target.target_span)
            theta_max = math.atan(S_target / (2 * z))

        # Compute pixel limit: p_limit = lambda / (2 * sin(theta_max))
        sin_theta = math.sin(theta_max)
        if sin_theta <= 0:
            return 1

        p_limit = wavelength / (2 * sin_theta)
        N_max = int(p_limit / p_global)

        return max(1, N_max)

    @property
    def prop_model(self) -> PropModel:
        """Get the determined propagation model."""
        return self._prop_model

    @property
    def slm_resolution(self) -> Tuple[int, int]:
        """Get SLM/DOE resolution in pixels."""
        return self._slm_resolution

    @property
    def phase_resolution(self) -> Tuple[int, int]:
        """Get resolution for phase optimization (may be downsampled)."""
        m = self.optimization.phase_pixel_multiplier
        return (self._slm_resolution[0] // m, self._slm_resolution[1] // m)

    @property
    def phase_pixel_size(self) -> Tuple[float, float]:
        """Get pixel size for phase optimization."""
        m = self.optimization.phase_pixel_multiplier
        ps = self.device.pixel_size
        return (ps * m, ps * m)

    @property
    def tolerance_limit(self) -> float:
        """Get computed tolerance limit (physical minimum)."""
        return self._tolerance_limit

    @property
    def max_pixel_multiplier(self) -> int:
        """Get maximum allowed pixel multiplier."""
        return self._max_pixel_multiplier

    def get_pixel_multiplier_options(self) -> List[int]:
        """Get list of valid pixel multiplier options for user selection."""
        return list(range(1, self._max_pixel_multiplier + 1))

    def get_feature_size(self, for_phase: bool = False) -> Tuple[float, float]:
        """Get feature size (pixel size) for optimization.

        Args:
            for_phase: If True, return pixel size for phase optimization (may be larger)
        """
        if for_phase:
            return self.phase_pixel_size
        return (self.device.pixel_size, self.device.pixel_size)

    def get_splitter_resolution(self) -> Optional[Tuple[int, int]]:
        """Get the optimization resolution for splitter DOE types.

        For splitters, the optimization resolution equals the period size in pixels.
        This ensures that:
        1. Each pixel in the optimization corresponds to one physical pixel
        2. FFT outputs have the same number of diffraction orders as period_pixels
        3. No non-integer resampling is needed between optimization and device phase

        The target pattern also has this resolution, with working orders marked
        at their correct positions within the k-space grid.

        Returns:
            (H, W) resolution for splitter optimization, or None if not a splitter
        """
        if not self.is_splitter():
            return None

        # Get period in pixels - this is the optimization resolution
        period = self.get_splitter_period()
        if period is None:
            return None

        period_pixels = int(round(period / self.device.pixel_size))

        # For 1D splitters, only use period_pixels in the y direction
        if self.doe_type == DOEType.SPLITTER_1D:
            return (period_pixels, 1)

        return (period_pixels, period_pixels)

    def is_splitter(self) -> bool:
        """Check if this is a splitter DOE type."""
        return self.doe_type in [DOEType.SPLITTER_1D, DOEType.SPLITTER_2D]

    def get_splitter_mode(self) -> Optional[SplitterMode]:
        """Get the splitter mode if this is a splitter DOE type."""
        if not self.is_splitter():
            return None
        mode_str = self.target.splitter_mode.lower()
        if mode_str == "uniform":
            return SplitterMode.UNIFORM_GRID
        return SplitterMode.NATURAL_GRID

    def get_finite_distance_strategy(self) -> Optional[FiniteDistanceStrategy]:
        """Determine the strategy for finite working distance splitter.

        Returns None if working distance is infinite.

        Strategy selection:
        - Strategy 1 (ASM): When target size is similar to DOE size, meaning the
          required simulation resolution is within MAX_OPTIMIZATION_RESOLUTION.
        - Strategy 2 (PERIODIC_FRESNEL): When target size is much larger, requiring
          periodic treatment + Fresnel phase overlay.
        """
        if self.physical.working_distance is None:
            return None  # Infinite distance, no strategy needed

        if not self.is_splitter():
            return None

        z = np.min(self.physical._working_distance_arr)
        target_size = max(self.target.target_span)
        pixel_size = self.device.pixel_size

        # Calculate required resolution for direct ASM approach
        # Target plane needs same pixel size as DOE plane in ASM
        required_pixels = int(target_size / pixel_size)

        if required_pixels <= MAX_OPTIMIZATION_RESOLUTION:
            return FiniteDistanceStrategy.ASM
        else:
            return FiniteDistanceStrategy.PERIODIC_FRESNEL

    def get_splitter_period(self) -> Optional[float]:
        """Compute the optimal period for splitter optimization.

        The period determines the k-space sampling grid and affects:
        - For NATURAL_GRID: period is set so diffraction orders exactly cover target span
        - For UNIFORM_GRID: period is set small enough to achieve tolerance when snapping

        For even spot counts with NATURAL_GRID, the period is doubled to allow
        symmetric order selection (skip alternate orders).

        Returns:
            Optimal period in meters, or None if not a splitter
        """
        if not self.is_splitter():
            return None

        # For finite distance Strategy 1 (ASM), return full device size (non-periodic)
        strategy = self.get_finite_distance_strategy()
        if strategy == FiniteDistanceStrategy.ASM:
            # Non-periodic optimization uses full device
            return self.device.diameter

        wavelength = self.physical._wavelength_arr.mean()
        mode = self.get_splitter_mode()
        num_spots = self.target.num_spots or (5, 5)

        if self.doe_type == DOEType.SPLITTER_1D:
            n_spots_y, n_spots_x = num_spots[0], 1
        else:
            n_spots_y, n_spots_x = num_spots

        n_spots = max(n_spots_y, n_spots_x)  # Use larger dimension for period calculation

        # Check if we have even spot counts (need special handling)
        is_even_y = (n_spots_y % 2 == 0)
        is_even_x = (n_spots_x % 2 == 0)
        has_even = is_even_y or is_even_x

        # Get target angle span (use half-angle for calculation)
        if self.target.target_type == "angle":
            # target_span is total FOV in radians
            theta_max = max(self.target.target_span) / 2
            sin_theta_max = math.sin(theta_max)
        else:
            # target_span is physical size, need to convert to angle
            z = self.physical.working_distance
            if z is None:
                raise ValueError("Cannot use 'size' target_type with infinite working distance")
            # For Strategy 2: treat as infinite distance
            S = max(self.target.target_span)
            if z is not None:
                z_val = np.min(np.atleast_1d(np.array(z)))
                theta_max = math.atan(S / (2 * z_val))
            else:
                theta_max = 0.1  # Default
            sin_theta_max = math.sin(theta_max)

        if mode == SplitterMode.NATURAL_GRID:
            # For natural grid: place n_spots at natural diffraction orders
            # Maximum order index for n_spots spots (centered): m_max = (n_spots - 1) / 2
            m_max = (n_spots - 1) / 2
            if m_max <= 0:
                m_max = 0.5  # At least 0th order

            # sin(theta_m) = m * lambda / period
            # For max order at theta_max: period = m_max * lambda / sin(theta_max)
            if sin_theta_max > 0:
                period = m_max * wavelength / sin_theta_max
            else:
                period = self.device.diameter  # Use full device if no span

            # For even spot counts: double the period to enable symmetric selection
            # This creates a finer k-space grid where we skip alternate orders
            if has_even:
                period = period * 2

            # Ensure period doesn't exceed device size
            period = min(period, self.device.diameter)

        else:  # UNIFORM_GRID
            # For uniform grid: need fine enough k-space sampling
            # to approximate uniform angle grid within tolerance
            # P >= lambda / (2 * T% * delta_sin_theta)
            tolerance = self.target.tolerance
            delta_sin_theta = 2 * sin_theta_max  # Full FOV in sin(theta)

            if delta_sin_theta > 0 and tolerance > 0:
                period_min = wavelength / (2 * tolerance * delta_sin_theta)
            else:
                period_min = self.device.diameter

            # Period should not exceed device size
            period = min(period_min, self.device.diameter)

            # For uniform grid, we also need to ensure we have enough resolution
            # to represent the snapped positions properly

        # Quantize period to be an integer multiple of pixel size
        pixel_size = self.device.pixel_size
        period_pixels = int(round(period / pixel_size))
        period = period_pixels * pixel_size

        return period

    def get_splitter_params(self) -> Optional[dict]:
        """Get complete splitter parameters including order angles.

        Returns dict with:
            - mode: SplitterMode
            - period: float (meters)
            - num_orders: Tuple[int, int] (total orders in y, x)
            - working_orders: List[Tuple[int, int]] (order indices)
            - order_angles: List[Tuple[float, float]] (angles in radians)
            - order_positions: List[Tuple[int, int]] (pixel positions in target)
            - finite_distance_strategy: FiniteDistanceStrategy or None
            - target_positions: List[Tuple[float, float]] (positions in meters, for finite distance)
            - is_periodic: bool (whether optimization uses periodic boundary)
        """
        if not self.is_splitter():
            return None

        mode = self.get_splitter_mode()
        period = self.get_splitter_period()
        wavelength = self.physical._wavelength_arr.mean()
        num_spots = self.target.num_spots or (5, 5)
        strategy = self.get_finite_distance_strategy()

        if self.doe_type == DOEType.SPLITTER_1D:
            n_rows, n_cols = num_spots[0], 1
        else:
            n_rows, n_cols = num_spots

        # Check for even spot counts (need symmetric order selection)
        is_even_y = (n_rows % 2 == 0)
        is_even_x = (n_cols % 2 == 0)

        # Determine if this is periodic optimization
        is_periodic = (
            self.physical.working_distance is None or
            strategy == FiniteDistanceStrategy.PERIODIC_FRESNEL
        )

        # Get working distance for finite case
        z = None
        if self.physical.working_distance is not None:
            z = np.min(self.physical._working_distance_arr)

        # Get target angle span
        if self.target.target_type == "angle":
            theta_span_y = self.target.target_span[1] if len(self.target.target_span) > 1 else self.target.target_span[0]
            theta_span_x = self.target.target_span[0]
        else:
            if z is None:
                raise ValueError("Cannot use 'size' target_type with infinite working distance")
            span_y = self.target.target_span[1] if len(self.target.target_span) > 1 else self.target.target_span[0]
            span_x = self.target.target_span[0]
            theta_span_y = 2 * math.atan(span_y / (2 * z))
            theta_span_x = 2 * math.atan(span_x / (2 * z))

        # Generate working order indices (centered around zero)
        working_orders = []
        order_angles = []

        if mode == SplitterMode.NATURAL_GRID:
            # Natural grid: orders are at natural diffraction angles
            # sin(theta_m) = m * lambda / period
            # For even counts with doubled period, select every other order for symmetry

            if is_even_y or is_even_x:
                # Even spot count: doubled period, skip alternate orders
                # Generate symmetric orders: ±1, ±3, ±5, ... for n_spots orders
                for iy in range(n_rows):
                    for ix in range(n_cols):
                        if self.doe_type == DOEType.SPLITTER_1D and ix != 0:
                            continue

                        # For even count, orders are: -(n-1), -(n-3), ..., -1, +1, ..., +(n-1)
                        # This is equivalent to: 2*i - (n-1) for i in range(n)
                        if is_even_y:
                            oy = 2 * iy - (n_rows - 1)
                        else:
                            oy = iy - n_rows // 2

                        if is_even_x:
                            ox = 2 * ix - (n_cols - 1)
                        else:
                            ox = ix - n_cols // 2

                        # Calculate angle from diffraction equation
                        sin_theta_y = oy * wavelength / period if period > 0 else 0
                        sin_theta_x = ox * wavelength / period if period > 0 else 0
                        # Clamp to valid range
                        sin_theta_y = max(-1, min(1, sin_theta_y))
                        sin_theta_x = max(-1, min(1, sin_theta_x))
                        theta_y = math.asin(sin_theta_y)
                        theta_x = math.asin(sin_theta_x)

                        working_orders.append((oy, ox))
                        order_angles.append((theta_y, theta_x))
            else:
                # Odd spot count: normal consecutive orders
                for oy in range(-(n_rows // 2), n_rows // 2 + 1):
                    if len(working_orders) // (n_cols if n_cols > 1 else 1) >= n_rows:
                        break
                    for ox in range(-(n_cols // 2), n_cols // 2 + 1):
                        if self.doe_type == DOEType.SPLITTER_1D and ox != 0:
                            continue
                        if len([w for w in working_orders if w[0] == oy]) >= n_cols:
                            break

                        # Calculate angle from diffraction equation
                        sin_theta_y = oy * wavelength / period if period > 0 else 0
                        sin_theta_x = ox * wavelength / period if period > 0 else 0
                        # Clamp to valid range
                        sin_theta_y = max(-1, min(1, sin_theta_y))
                        sin_theta_x = max(-1, min(1, sin_theta_x))
                        theta_y = math.asin(sin_theta_y)
                        theta_x = math.asin(sin_theta_x)

                        working_orders.append((oy, ox))
                        order_angles.append((theta_y, theta_x))

        else:  # UNIFORM_GRID
            # Uniform grid: spots at uniform angle spacing, snapped to k-space grid
            # Edge spots should be at ±span/2, so step = span / (n-1)
            angle_step_y = theta_span_y / (n_rows - 1) if n_rows > 1 else 0
            angle_step_x = theta_span_x / (n_cols - 1) if n_cols > 1 else 0

            for iy in range(n_rows):
                for ix in range(n_cols):
                    if self.doe_type == DOEType.SPLITTER_1D and ix != 0:
                        continue

                    # Target uniform angle (centered)
                    target_theta_y = (iy - (n_rows - 1) / 2) * angle_step_y
                    target_theta_x = (ix - (n_cols - 1) / 2) * angle_step_x

                    # Convert to k-space and snap to grid
                    target_sin_y = math.sin(target_theta_y)
                    target_sin_x = math.sin(target_theta_x)

                    # k-space grid: sin(theta) = m * lambda / period
                    # m = sin(theta) * period / lambda
                    order_y = round(target_sin_y * period / wavelength) if period > 0 else 0
                    order_x = round(target_sin_x * period / wavelength) if period > 0 else 0

                    # Actual angle after snapping
                    actual_sin_y = order_y * wavelength / period if period > 0 else 0
                    actual_sin_x = order_x * wavelength / period if period > 0 else 0
                    actual_sin_y = max(-1, min(1, actual_sin_y))
                    actual_sin_x = max(-1, min(1, actual_sin_x))
                    actual_theta_y = math.asin(actual_sin_y)
                    actual_theta_x = math.asin(actual_sin_x)

                    working_orders.append((order_y, order_x))
                    order_angles.append((actual_theta_y, actual_theta_x))

        # Calculate period in pixels - this determines the FFT grid size
        # and therefore the optimization/target pattern resolution
        pixel_size = self.device.pixel_size
        period_pixels = int(round(period / pixel_size))

        # The target pattern resolution equals period_pixels
        # This ensures 1:1 correspondence between optimization pixels and device pixels
        # (no non-integer resampling needed)
        tot_orders_y = period_pixels
        tot_orders_x = period_pixels if n_cols > 1 else 1

        # Calculate pixel positions in target pattern
        # Working orders are placed at their natural positions in the k-space grid
        order_positions = []
        for oy, ox in working_orders:
            py = period_pixels // 2 + oy
            # For 1D splitters, x is always 0 (only one column)
            if self.doe_type == DOEType.SPLITTER_1D:
                px = 0
            else:
                px = period_pixels // 2 + ox
            order_positions.append((py, px))

        # Calculate physical target positions for finite distance
        target_positions = []
        if z is not None:
            if strategy == FiniteDistanceStrategy.ASM:
                # For ASM (Strategy 1): place spots directly at physical positions
                # based on target_span, not diffraction angles
                if self.target.target_type == "size":
                    span_y = self.target.target_span[1] if len(self.target.target_span) > 1 else self.target.target_span[0]
                    span_x = self.target.target_span[0]
                else:
                    # For angle target type, compute physical span
                    span_y = 2 * z * math.tan(theta_span_y / 2)
                    span_x = 2 * z * math.tan(theta_span_x / 2)

                # Uniform grid of spots across the target span
                for iy in range(n_rows):
                    for ix in range(n_cols):
                        # Position from -span/2 to +span/2
                        if n_rows > 1:
                            pos_y = -span_y / 2 + iy * span_y / (n_rows - 1)
                        else:
                            pos_y = 0.0
                        if n_cols > 1:
                            pos_x = -span_x / 2 + ix * span_x / (n_cols - 1)
                        else:
                            pos_x = 0.0
                        target_positions.append((pos_y, pos_x))
            else:
                # For periodic strategies: compute from diffraction angles
                for theta_y, theta_x in order_angles:
                    pos_y = z * math.tan(theta_y)
                    pos_x = z * math.tan(theta_x)
                    target_positions.append((pos_y, pos_x))

        return {
            'mode': mode,
            'period': period,
            'wavelength': wavelength,
            'num_orders': (tot_orders_y, tot_orders_x),
            'working_orders': working_orders,
            'order_angles': order_angles,
            'order_positions': order_positions,
            'target_span': (theta_span_y, theta_span_x),
            'finite_distance_strategy': strategy,
            'target_positions': target_positions if target_positions else None,
            'is_periodic': is_periodic,
            'is_even_spots': (is_even_y, is_even_x),
            'working_distance': z,
        }
