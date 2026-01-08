"""
Splitter wizard for 1D and 2D beam splitters.

Handles:
- Natural vs Uniform grid mode period calculation
- Even spot count symmetry handling
- Finite distance strategy selection (ASM vs Periodic+Fresnel)
- Order angle computation
- Target pattern generation
"""

import math
from typing import Dict, Any, List, Tuple, Optional
import torch
import numpy as np

from .base import BaseWizard, WizardOutput
from ..params.base import PhysicalConstants, PropagatorConfig, PropagationType
from ..params.fft_params import FFTParams
from ..params.asm_params import ASMParams
from ..params.sfr_params import SFRParams
from ..params.optimization import OptimizationConfig
from ..api.errors import ValidationWarning, WarningCode


class FiniteDistanceStrategy:
    """Strategy for finite distance splitter optimization."""
    ASM = "asm"              # Strategy 1: Direct ASM for small targets
    SFR = "sfr"              # Strategy 1: Direct SFR for large targets
    PERIODIC_FRESNEL = "periodic_fresnel"  # Strategy 2: Periodic + Fresnel


class SplitterMode:
    """Splitter grid mode."""
    NATURAL = "natural"
    UNIFORM = "uniform"


class SplitterWizard(BaseWizard):
    """Wizard for 1D and 2D splitter parameter generation.

    Handles both infinite and finite distance splitters with:
    - Natural grid mode (diffraction order aligned)
    - Uniform grid mode (angle space uniform with tolerance)
    - Automatic strategy selection for finite distance
    - Even spot count symmetry handling
    """

    # Margin factor for ASM/SFR target area to avoid edge truncation
    # 1.1 = 10% larger simulation area than target pattern
    TARGET_MARGIN_FACTOR = 1.1

    def __init__(self, is_1d: bool = False, max_resolution: int = 2000,
                 target_margin_factor: float = None):
        """Initialize splitter wizard.

        Args:
            is_1d: True for 1D splitter, False for 2D
            max_resolution: Maximum simulation resolution
            target_margin_factor: Margin factor for ASM/SFR target area (default 1.1)
        """
        super().__init__(max_resolution)
        self.is_1d = is_1d
        self.target_margin_factor = target_margin_factor or self.TARGET_MARGIN_FACTOR

    def generate_params(
        self,
        user_input: Dict[str, Any],
        device: torch.device = None
    ) -> WizardOutput:
        """Generate splitter parameters from user input.

        Args:
            user_input: User input containing:
                - wavelength: float (meters)
                - working_distance: float or None (meters)
                - device_diameter: float (meters)
                - pixel_size: float (meters)
                - target_spec: dict with:
                    - num_spots: int or [ny, nx]
                    - target_type: 'angle' or 'size'
                    - target_span: float or [span_y, span_x]
                    - grid_mode: 'natural' or 'uniform'
                    - tolerance: float (for uniform mode)
                    - use_strategy2: bool (optional, for finite distance)
                - optimization: dict (optional)
            device: Torch device

        Returns:
            WizardOutput with splitter parameters
        """
        if device is not None:
            self.device = device

        # Get target_margin from user input (from advanced or optimization settings)
        # Frontend sends target_margin as decimal (e.g., 0.1 for 10%)
        advanced_settings = user_input.get('advanced', {})
        opt_settings = user_input.get('optimization', {})
        target_margin = advanced_settings.get('target_margin') or opt_settings.get('target_margin')
        if target_margin is not None:
            self.target_margin_factor = 1.0 + target_margin

        # Extract basic parameters
        wavelength = user_input['wavelength']
        working_distance = user_input.get('working_distance')
        device_diameter = user_input['device_diameter']
        pixel_size = user_input['pixel_size']
        refraction_index = user_input.get('refraction_index', 1.62)

        # Extract target spec
        target_spec = user_input['target_spec']
        num_spots = target_spec['num_spots']
        if isinstance(num_spots, int):
            num_spots = (num_spots, 1) if self.is_1d else (num_spots, num_spots)
        num_spots = tuple(num_spots)

        target_type = target_spec.get('target_type', 'angle')
        target_span = target_spec['target_span']
        if isinstance(target_span, (int, float)):
            target_span = (target_span, target_span)
        target_span = tuple(target_span)

        grid_mode = target_spec.get('grid_mode', 'natural')
        tolerance = target_spec.get('tolerance', 0.05)
        use_strategy2 = target_spec.get('use_strategy2', False)

        # Create physical constants
        physical = PhysicalConstants(
            wavelength=wavelength,
            refraction_index=refraction_index,
            pixel_size=pixel_size
        )

        # Compute device pixels
        device_pixels = int(round(device_diameter / pixel_size))

        # Collect warnings
        warnings = []
        self._validate_sampling(wavelength, pixel_size, warnings)

        # Initialize physical_positions (will be set for finite distance strategies)
        physical_positions = None

        # Determine finite distance strategy
        # Respect user's explicit use_strategy2 choice
        strategy = self._determine_strategy(
            working_distance=working_distance,
            target_span=target_span,
            pixel_size=pixel_size,
            target_type=target_type,
            device_diameter=device_diameter,
            use_strategy2=use_strategy2
        )

        # Convert target span to angles if needed
        if target_type == 'size':
            if working_distance is None:
                raise ValueError("Cannot use 'size' target_type with infinite distance")
            theta_span = tuple(
                2 * math.atan(s / (2 * working_distance))
                for s in target_span
            )
        else:
            theta_span = target_span

        # Compute period
        period = self._compute_period(
            wavelength=wavelength,
            num_spots=num_spots,
            theta_span=theta_span,
            grid_mode=grid_mode,
            tolerance=tolerance,
            device_diameter=device_diameter,
            strategy=strategy
        )

        period_pixels = int(round(period / pixel_size))

        # Compute working orders and angles
        working_orders, order_angles = self._compute_working_orders(
            wavelength=wavelength,
            period=period,
            num_spots=num_spots,
            grid_mode=grid_mode,
            theta_span=theta_span
        )

        # Create structured params based on strategy
        if strategy == FiniteDistanceStrategy.ASM:
            # Strategy 1: Direct ASM optimization (small target, physical space)
            structured_params = self._create_asm_params(
                physical=physical,
                device_pixels=device_pixels,
                working_distance=working_distance,
                target_span=target_span,
                target_type=target_type
            )
            prop_type = PropagationType.ASM

            # Target pattern for ASM: spots at physical positions (NOT diffraction orders)
            target_pattern, physical_positions = self._generate_physical_space_target(
                output_pixels=structured_params.target_pixels,
                pixel_size=pixel_size,
                num_spots=num_spots,
                target_span=target_span if target_type == 'size' else None,
                theta_span=theta_span if target_type == 'angle' else None,
                working_distance=working_distance,
                wavelength=wavelength,
                doe_size=device_diameter
            )
            # Override order_angles with physical positions for ASM
            order_angles = self._compute_physical_angles(physical_positions, working_distance)

        elif strategy == FiniteDistanceStrategy.SFR:
            # Strategy 1: Direct SFR optimization (large target, physical space)
            structured_params = self._create_sfr_params(
                physical=physical,
                device_pixels=device_pixels,
                working_distance=working_distance,
                target_span=target_span,
                target_type=target_type
            )
            prop_type = PropagationType.SFR

            # Target pattern for SFR: spots at physical positions
            target_pattern, physical_positions = self._generate_physical_space_target(
                output_pixels=structured_params.target_resolution,
                pixel_size=structured_params.target_pixel_size[0],
                num_spots=num_spots,
                target_span=target_span if target_type == 'size' else None,
                theta_span=theta_span if target_type == 'angle' else None,
                working_distance=working_distance,
                wavelength=wavelength,
                doe_size=device_diameter
            )
            order_angles = self._compute_physical_angles(physical_positions, working_distance)

        else:
            # Strategy 2 (periodic_fresnel) or Infinite: FFT optimization (k-space)
            structured_params = self._create_fft_params(
                physical=physical,
                period_pixels=period_pixels,
                device_pixels=device_pixels
            )
            prop_type = PropagationType.FFT

            # Target pattern for FFT: delta functions at order positions in k-space
            target_pattern = self._generate_fft_target(
                period_pixels=period_pixels,
                working_orders=working_orders,
                period=period,
                wavelength=wavelength
            )

            # For Strategy 2 (periodic_fresnel with finite distance), compute physical positions
            # This allows the preview to show physical coordinates instead of angles
            if strategy == FiniteDistanceStrategy.PERIODIC_FRESNEL and working_distance is not None:
                physical_positions = self._compute_physical_positions_from_angles(
                    order_angles, working_distance
                )

        # Create propagator config
        propagator_config = structured_params.to_propagator_config()

        # Create optimization config
        optimization_config = self._create_optimization_config(user_input)

        # Compute derived values for frontend
        tolerance_limit = self._compute_tolerance_limit(
            wavelength=wavelength,
            period=period,
            target_span=max(theta_span)
        )

        max_multiplier = self._compute_max_pixel_multiplier(
            base_pixels=period_pixels,
            device_pixels=device_pixels
        )

        # Count non-zero pixels in target pattern (= number of spots/orders)
        num_orders = int((target_pattern > 0).sum().item())

        computed_values = {
            'period': period,
            'period_pixels': period_pixels,
            'tolerance_limit': tolerance_limit,
            'max_pixel_multiplier': max_multiplier,
            'working_orders': working_orders,
            'order_angles': order_angles,
            'strategy': strategy,
            'num_periods': (device_pixels // period_pixels, device_pixels // period_pixels),
            'target_margin_factor': self.target_margin_factor,
            'num_orders': num_orders,
        }

        # Compute order positions for visualization
        # For ASM/SFR, positions are in physical space; for FFT, positions are in k-space
        if strategy == FiniteDistanceStrategy.ASM:
            order_positions = self._compute_physical_order_positions(
                output_pixels=structured_params.target_pixels,
                pixel_size=pixel_size,
                physical_positions=physical_positions
            )
        elif strategy == FiniteDistanceStrategy.SFR:
            order_positions = self._compute_physical_order_positions(
                output_pixels=structured_params.target_resolution,
                pixel_size=structured_params.target_pixel_size[0],
                physical_positions=physical_positions
            )
        else:
            order_positions = self._compute_order_positions(
                period_pixels=period_pixels,
                working_orders=working_orders,
                is_1d=self.is_1d
            )

        # Splitter-specific metadata
        # physical_positions may be set by ASM/SFR target generation or Strategy 2 conversion
        metadata = {
            'is_1d': self.is_1d,
            'grid_mode': grid_mode,
            'num_spots': num_spots,
            'is_even_y': num_spots[0] % 2 == 0,
            'is_even_x': num_spots[1] % 2 == 0,
            'target_type': target_type,
            'theta_span': theta_span,
            'target_span': target_span if target_type == 'size' else None,
            'working_orders': working_orders,
            'order_angles': order_angles,
            'order_positions': order_positions,
            'physical_positions': physical_positions,
            'period_pixels': period_pixels,
            'wavelength': wavelength,
            'working_distance': working_distance,
            'strategy': strategy,
            'target_margin_factor': self.target_margin_factor,
        }

        return WizardOutput(
            structured_params=structured_params,
            propagator_config=propagator_config,
            optimization_config=optimization_config,
            target_pattern=target_pattern,
            computed_values=computed_values,
            warnings=warnings,
            metadata=metadata
        )

    def get_constraints(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Get constraints for frontend validation."""
        wavelength = user_input.get('wavelength', 532e-9)
        pixel_size = user_input.get('pixel_size', 0.5e-6)
        device_diameter = user_input.get('device_diameter', 1e-3)

        # Maximum angle without aliasing
        max_sin = wavelength / (2 * pixel_size)
        max_angle = math.asin(min(1, max_sin)) if max_sin < 1 else math.pi / 2

        return {
            'num_spots': {
                'min': 1,
                'max': 100,
            },
            'target_span': {
                'min_angle': 0.001,  # ~0.06 degrees
                'max_angle': 2 * max_angle,
            },
            'tolerance': {
                'min': 0.001,
                'max': 0.5,
                'default': 0.05,
            },
            'max_angle_degrees': math.degrees(max_angle),
        }

    def _determine_strategy(
        self,
        working_distance: Optional[float],
        target_span: Tuple[float, float],
        pixel_size: float,
        target_type: str,
        device_diameter: float,
        use_strategy2: bool = False
    ) -> Optional[str]:
        """Determine finite distance strategy.

        Strategy 1 (direct propagation) - when use_strategy2 is False:
        - ASM: For small targets (target_size <= DOE_size * 2)
        - SFR: For large targets (target_size > DOE_size * 2)

        Strategy 2 (periodic + Fresnel) - when use_strategy2 is True:
        - Uses periodic optimization with Fresnel lens overlay for evaluation

        The user's explicit choice of use_strategy2 is respected.
        When use_strategy2 is False, only ASM or SFR will be used.

        Returns:
            'asm', 'sfr', 'periodic_fresnel', or None (infinite distance)
        """
        if working_distance is None:
            return None

        # Force FFT for angle-based targets even if working_distance is set
        # This handles the case where user switches from 'size' to 'angle' mode
        # (angle-based at finite distance without Strategy 2 doesn't make sense)
        if target_type == 'angle' and not use_strategy2:
            return None

        # If user explicitly chose Strategy 2, use periodic_fresnel
        if use_strategy2:
            return FiniteDistanceStrategy.PERIODIC_FRESNEL

        # Strategy 1: Direct propagation (ASM or SFR)
        # Compute target size
        if target_type == 'size':
            target_size = max(target_span)
        else:
            # For angle-based, estimate target size
            target_size = 2 * working_distance * math.tan(max(target_span) / 2)

        required_pixels = int(target_size / pixel_size)

        # ASM can handle targets up to ~2x DOE size (with linear convolution padding)
        # Beyond that, use SFR with zoom-FFT
        asm_limit = device_diameter * 2

        if target_size <= asm_limit and required_pixels <= self.max_resolution:
            # Small target: use ASM (Strategy 1)
            return FiniteDistanceStrategy.ASM
        else:
            # Large target: use SFR (Strategy 1)
            # SFR uses zoom-FFT so can handle any target size with fixed DOE resolution
            return FiniteDistanceStrategy.SFR

    def _compute_period(
        self,
        wavelength: float,
        num_spots: Tuple[int, int],
        theta_span: Tuple[float, float],
        grid_mode: str,
        tolerance: float,
        device_diameter: float,
        strategy: Optional[str]
    ) -> float:
        """Compute optimal period for splitter.

        For ASM/SFR strategy (non-periodic), returns device diameter.
        For FFT/periodic_fresnel, computes period based on grid mode.

        Note: Grid Mode (natural/uniform) only applies to periodic cases (FFT/periodic_fresnel).
        For ASM/SFR, spots are uniformly distributed in physical space regardless of grid_mode.
        """
        # For ASM or SFR (Strategy 1), use full device (non-periodic)
        if strategy in (FiniteDistanceStrategy.ASM, FiniteDistanceStrategy.SFR):
            return device_diameter

        n_spots = max(num_spots)
        theta_max = max(theta_span) / 2
        sin_theta_max = math.sin(theta_max)

        # Check for even spot counts
        is_even = any(n % 2 == 0 for n in num_spots)

        if grid_mode == SplitterMode.NATURAL:
            # Natural grid: orders at natural diffraction angles
            m_max = (n_spots - 1) / 2
            if m_max <= 0:
                m_max = 0.5

            if sin_theta_max > 0:
                period = m_max * wavelength / sin_theta_max
            else:
                period = device_diameter

            # Double period for even spot counts (symmetric selection)
            if is_even:
                period = period * 2

        else:  # UNIFORM grid
            # Need fine k-space sampling for uniform angle snapping
            delta_sin_theta = 2 * sin_theta_max

            if delta_sin_theta > 0 and tolerance > 0:
                period = wavelength / (2 * tolerance * delta_sin_theta)
            else:
                period = device_diameter

        # Clamp to device size
        period = min(period, device_diameter)

        return period

    def _compute_working_orders(
        self,
        wavelength: float,
        period: float,
        num_spots: Tuple[int, int],
        grid_mode: str,
        theta_span: Tuple[float, float]
    ) -> Tuple[List[Tuple[int, int]], List[Tuple[float, float]]]:
        """Compute working order indices and their angles.

        Returns:
            (working_orders, order_angles) where each is a list of (y, x) tuples
        """
        n_rows, n_cols = num_spots
        if self.is_1d:
            n_cols = 1

        is_even_y = n_rows % 2 == 0
        is_even_x = n_cols % 2 == 0

        working_orders = []
        order_angles = []

        for iy in range(n_rows):
            for ix in range(n_cols):
                if self.is_1d and ix != 0:
                    continue

                if grid_mode == SplitterMode.NATURAL:
                    # For even counts with doubled period, skip alternate orders
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

                else:  # UNIFORM grid
                    # Uniformly spaced in angle, then snap to nearest order
                    theta_y = (iy / (n_rows - 1) - 0.5) * theta_span[0] if n_rows > 1 else 0
                    theta_x = (ix / (n_cols - 1) - 0.5) * theta_span[1] if n_cols > 1 else 0

                    sin_theta_y = math.sin(theta_y)
                    sin_theta_x = math.sin(theta_x)

                    # Snap to nearest order
                    oy = round(sin_theta_y * period / wavelength)
                    ox = round(sin_theta_x * period / wavelength)

                    # Recalculate actual angle after snapping
                    sin_theta_y = oy * wavelength / period if period > 0 else 0
                    sin_theta_x = ox * wavelength / period if period > 0 else 0

                # Clamp to valid range and convert to angle
                sin_theta_y = max(-1, min(1, sin_theta_y))
                sin_theta_x = max(-1, min(1, sin_theta_x))
                theta_y = math.asin(sin_theta_y)
                theta_x = math.asin(sin_theta_x)

                working_orders.append((oy, ox))
                order_angles.append((theta_y, theta_x))

        return working_orders, order_angles

    def _create_fft_params(
        self,
        physical: PhysicalConstants,
        period_pixels: int,
        device_pixels: int
    ) -> FFTParams:
        """Create FFT parameters for periodic optimization."""
        return FFTParams(
            period_pixels=(period_pixels, 1 if self.is_1d else period_pixels),
            doe_total_pixels=(device_pixels, device_pixels),
            physical=physical,
            upsample_factor=1,
            aperture_type='square'
        )

    def _create_asm_params(
        self,
        physical: PhysicalConstants,
        device_pixels: int,
        working_distance: float,
        target_span: Tuple[float, float],
        target_type: str
    ) -> ASMParams:
        """Create ASM parameters for non-periodic optimization.

        Applies margin factor to target size to avoid edge truncation.
        """
        # Target size determines output resolution
        if target_type == 'size':
            target_size = max(target_span)
        else:
            target_size = 2 * working_distance * math.tan(max(target_span) / 2)

        # Apply margin factor to avoid edge truncation
        target_size_with_margin = target_size * self.target_margin_factor

        target_pixels = int(target_size_with_margin / physical.pixel_size)
        # Ensure at least as large as DOE
        target_pixels = max(target_pixels, device_pixels)

        return ASMParams(
            doe_pixels=(device_pixels, device_pixels),
            physical=physical,
            working_distances=[working_distance],
            target_pixels=(target_pixels, target_pixels),
            upsample_factor=1,
            aperture_type='square'
        )

    def _generate_fft_target(
        self,
        period_pixels: int,
        working_orders: List[Tuple[int, int]],
        period: float,
        wavelength: float
    ) -> torch.Tensor:
        """Generate target pattern for FFT optimization.

        Creates delta functions at the order positions in k-space.
        """
        h = period_pixels
        w = 1 if self.is_1d else period_pixels

        target = torch.zeros(1, 1, h, w, device=self.device, dtype=self.dtype)

        for oy, ox in working_orders:
            # Order position in FFT output (centered at h//2, w//2)
            # FFT output: order m is at position (h//2 + m, w//2 + m)
            py = (h // 2 + oy) % h
            px = (w // 2 + ox) % w if not self.is_1d else 0

            target[0, 0, py, px] = 1.0

        # Normalize to unit total energy
        target = target / (target.sum() + 1e-10).sqrt()

        return target

    def _compute_order_positions(
        self,
        period_pixels: int,
        working_orders: List[Tuple[int, int]],
        is_1d: bool
    ) -> List[Tuple[int, int]]:
        """Compute pixel positions of orders in FFT output.

        Args:
            period_pixels: Size of the FFT output
            working_orders: List of (oy, ox) order indices
            is_1d: Whether this is a 1D splitter

        Returns:
            List of (py, px) pixel positions in FFT output
        """
        h = period_pixels
        w = 1 if is_1d else period_pixels

        positions = []
        for oy, ox in working_orders:
            # Order position in FFT output (centered at h//2, w//2)
            py = (h // 2 + oy) % h
            px = (w // 2 + ox) % w if not is_1d else 0
            positions.append((py, px))

        return positions

    def _compute_asm_order_positions(
        self,
        structured_params: ASMParams,
        order_angles: List[Tuple[float, float]],
        working_distance: float
    ) -> List[Tuple[int, int]]:
        """Compute pixel positions of orders in ASM output (physical space).

        For ASM propagation, spots are at physical positions on the target plane,
        not at k-space positions like FFT.

        Args:
            structured_params: ASM parameters with target pixel info
            order_angles: List of (theta_y, theta_x) angles for each order
            working_distance: Propagation distance

        Returns:
            List of (py, px) pixel positions in ASM output
        """
        h, w = structured_params.target_pixels
        pixel_size = structured_params.physical.pixel_size

        # Center of target
        cy, cx = h // 2, w // 2

        positions = []
        for (theta_y, theta_x) in order_angles:
            # Physical position from angle
            py_m = working_distance * math.tan(theta_y)
            px_m = working_distance * math.tan(theta_x)

            # Convert to pixel position (centered)
            py = int(cy + py_m / pixel_size)
            px = int(cx + px_m / pixel_size)

            # Clamp to valid range
            py = max(0, min(h - 1, py))
            px = max(0, min(w - 1, px))

            positions.append((py, px))

        return positions

    def _generate_asm_target(
        self,
        structured_params: ASMParams,
        working_orders: List[Tuple[int, int]],
        order_angles: List[Tuple[float, float]],
        working_distance: float
    ) -> torch.Tensor:
        """Generate target pattern for ASM optimization.

        Creates Gaussian spots at physical positions.
        """
        h, w = structured_params.target_pixels
        pixel_size = structured_params.physical.pixel_size
        wavelength = structured_params.physical.wavelength

        target = torch.zeros(1, 1, h, w, device=self.device, dtype=self.dtype)

        # Airy disk radius (spot size)
        doe_size = structured_params.doe_pixels[0] * pixel_size
        airy_radius = 1.22 * wavelength * working_distance / doe_size
        spot_sigma = airy_radius / pixel_size / 2.355  # Convert to sigma in pixels

        # Center of target
        cy, cx = h // 2, w // 2

        for (theta_y, theta_x) in order_angles:
            # Physical position
            py_m = working_distance * math.tan(theta_y)
            px_m = working_distance * math.tan(theta_x)

            # Pixel position
            py = int(cy + py_m / pixel_size)
            px = int(cx + px_m / pixel_size)

            # Add Gaussian spot
            if 0 <= py < h and 0 <= px < w:
                y_coords = torch.arange(h, device=self.device, dtype=self.dtype) - py
                x_coords = torch.arange(w, device=self.device, dtype=self.dtype) - px

                yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
                gaussian = torch.exp(-(yy**2 + xx**2) / (2 * spot_sigma**2))

                target[0, 0] += gaussian

        # Normalize
        target = target / (target.sum() + 1e-10).sqrt()

        return target

    def _create_sfr_params(
        self,
        physical: PhysicalConstants,
        device_pixels: int,
        working_distance: float,
        target_span: Tuple[float, float],
        target_type: str
    ) -> SFRParams:
        """Create SFR parameters for large target optimization.

        SFR (Scaled Fresnel) is used when target is larger than DOE,
        using zoom-FFT for arbitrary output size.

        Applies margin factor to target size to avoid edge truncation.
        """
        # Target size in meters
        if target_type == 'size':
            target_size_base = target_span
        else:
            target_size_base = tuple(
                2 * working_distance * math.tan(s / 2)
                for s in target_span
            )

        # Apply margin factor to avoid edge truncation
        target_size = tuple(s * self.target_margin_factor for s in target_size_base)

        # Target resolution - keep same as DOE pixels
        # SFR uses zoom-FFT which handles scaling internally
        # Using DOE pixels ensures simulation_pixels = doe_pixels consistently
        target_resolution = (device_pixels, device_pixels)

        return SFRParams(
            doe_pixels=(device_pixels, device_pixels),
            physical=physical,
            working_distances=[working_distance],
            target_size=target_size,
            target_resolution=target_resolution,
            upsample_factor=1,
            aperture_type='square'
        )

    def _generate_physical_space_target(
        self,
        output_pixels: Tuple[int, int],
        pixel_size: float,
        num_spots: Tuple[int, int],
        target_span: Optional[Tuple[float, float]],
        theta_span: Optional[Tuple[float, float]],
        working_distance: float,
        wavelength: float,
        doe_size: float
    ) -> Tuple[torch.Tensor, List[Tuple[float, float]]]:
        """Generate target pattern for physical space optimization (ASM/SFR).

        Places spots at physical positions based on target_span or theta_span,
        NOT using diffraction order calculations.

        Args:
            output_pixels: Output array size (H, W)
            pixel_size: Pixel size in output plane (meters)
            num_spots: Number of spots (ny, nx)
            target_span: Target physical size (meters), or None if using angles
            theta_span: Target angle span (radians), or None if using size
            working_distance: Propagation distance (meters)
            wavelength: Wavelength (meters)
            doe_size: DOE diameter (meters)

        Returns:
            (target_pattern, physical_positions) where physical_positions is
            list of (y_meters, x_meters) from center
        """
        h, w = output_pixels
        ny, nx = num_spots

        # Handle 1D case
        if self.is_1d:
            nx = 1

        target = torch.zeros(1, 1, h, w, device=self.device, dtype=self.dtype)

        # Compute physical extent of target
        if target_span is not None:
            span_y, span_x = target_span
        else:
            # Convert angle span to physical span
            span_y = 2 * working_distance * math.tan(theta_span[0] / 2)
            span_x = 2 * working_distance * math.tan(theta_span[1] / 2)

        # Airy disk radius for spot size
        airy_radius = 1.22 * wavelength * working_distance / doe_size
        spot_sigma = airy_radius / pixel_size / 2.355  # Sigma in pixels
        spot_sigma = max(spot_sigma, 1.0)  # Minimum 1 pixel

        # Center of output
        cy, cx = h // 2, w // 2

        # Compute spot positions (uniformly distributed over target span)
        physical_positions = []

        for iy in range(ny):
            for ix in range(nx):
                if self.is_1d and ix != 0:
                    continue

                # Physical position from center (uniformly spaced)
                if ny > 1:
                    pos_y = (iy / (ny - 1) - 0.5) * span_y
                else:
                    pos_y = 0.0

                if nx > 1:
                    pos_x = (ix / (nx - 1) - 0.5) * span_x
                else:
                    pos_x = 0.0

                physical_positions.append((pos_y, pos_x))

                # Convert to pixel position
                py = int(cy + pos_y / pixel_size)
                px = int(cx + pos_x / pixel_size)

                # Add Gaussian spot
                if 0 <= py < h and 0 <= px < w:
                    y_coords = torch.arange(h, device=self.device, dtype=self.dtype) - py
                    x_coords = torch.arange(w, device=self.device, dtype=self.dtype) - px

                    yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
                    gaussian = torch.exp(-(yy**2 + xx**2) / (2 * spot_sigma**2))

                    target[0, 0] += gaussian

        # Normalize
        target = target / (target.sum() + 1e-10).sqrt()

        return target, physical_positions

    def _compute_physical_angles(
        self,
        physical_positions: List[Tuple[float, float]],
        working_distance: float
    ) -> List[Tuple[float, float]]:
        """Compute angles from physical positions.

        Args:
            physical_positions: List of (y_meters, x_meters) from center
            working_distance: Propagation distance

        Returns:
            List of (theta_y, theta_x) angles in radians
        """
        angles = []
        for pos_y, pos_x in physical_positions:
            theta_y = math.atan(pos_y / working_distance) if working_distance > 0 else 0
            theta_x = math.atan(pos_x / working_distance) if working_distance > 0 else 0
            angles.append((theta_y, theta_x))
        return angles

    def _compute_physical_positions_from_angles(
        self,
        order_angles: List[Tuple[float, float]],
        working_distance: float
    ) -> List[Tuple[float, float]]:
        """Compute physical positions from angles.

        Inverse of _compute_physical_angles. Used for Strategy 2 to convert
        k-space angles to physical positions on target plane.

        Args:
            order_angles: List of (theta_y, theta_x) angles in radians
            working_distance: Propagation distance in meters

        Returns:
            List of (y_meters, x_meters) physical positions from center
        """
        positions = []
        for theta_y, theta_x in order_angles:
            pos_y = working_distance * math.tan(theta_y) if working_distance > 0 else 0
            pos_x = working_distance * math.tan(theta_x) if working_distance > 0 else 0
            positions.append((pos_y, pos_x))
        return positions

    def _compute_physical_order_positions(
        self,
        output_pixels: Tuple[int, int],
        pixel_size: float,
        physical_positions: List[Tuple[float, float]]
    ) -> List[Tuple[int, int]]:
        """Compute pixel positions from physical positions.

        Args:
            output_pixels: Output array size (H, W)
            pixel_size: Pixel size in output plane (meters)
            physical_positions: List of (y_meters, x_meters) from center

        Returns:
            List of (py, px) pixel positions
        """
        h, w = output_pixels
        cy, cx = h // 2, w // 2

        positions = []
        for pos_y, pos_x in physical_positions:
            py = int(cy + pos_y / pixel_size)
            px = int(cx + pos_x / pixel_size)

            # Clamp to valid range
            py = max(0, min(h - 1, py))
            px = max(0, min(w - 1, px))

            positions.append((py, px))

        return positions
