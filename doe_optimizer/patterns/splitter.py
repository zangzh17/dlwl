"""Beam splitter pattern generator."""

import numpy as np
import torch

from .base import PatternGenerator
from ..core.config import DOEConfig, DOEType, SplitterMode, FiniteDistanceStrategy


class SplitterGenerator(PatternGenerator):
    """Generate beam splitter (spot array) patterns.

    Supports both 1D and 2D splitters with two grid modes:
    - NATURAL_GRID: K-space uniform (natural diffraction orders)
    - UNIFORM_GRID: Angle/size space uniform (snapped to k-space grid)

    For splitters, the target pattern has a small resolution matching the
    number of diffraction orders, where each pixel represents one order.

    Exception: For Strategy 1 (ASM), the target is at full resolution with
    spots placed at physical positions in the output plane.
    """

    def generate(self) -> torch.Tensor:
        """Generate splitter target pattern.

        For periodic splitters (infinite distance or Strategy 2), we generate
        a small pattern where each pixel corresponds to one diffraction order.

        For Strategy 1 (ASM), we generate a full-resolution pattern with
        spots at physical positions.

        Returns:
            Target amplitude tensor [1, C, H, W]
        """
        params = self.config.get_splitter_params()
        if params is None:
            raise ValueError("Not a splitter DOE type")

        # Check if using Strategy 1 (ASM)
        strategy = self.config.get_finite_distance_strategy()
        if strategy == FiniteDistanceStrategy.ASM:
            return self._generate_asm_target(params)
        else:
            return self._generate_periodic_target(params)

    def _generate_periodic_target(self, params: dict) -> torch.Tensor:
        """Generate target for periodic optimization (FFT).

        Creates a small pattern where each pixel = one diffraction order.
        """
        tot_orders_y, tot_orders_x = params['num_orders']
        order_positions = params['order_positions']

        # Create pattern: each pixel = one diffraction order
        pattern = torch.zeros(
            1, self.num_channels, tot_orders_y, tot_orders_x,
            device=self.device
        )

        # Amplitude per spot (uniform energy distribution)
        n_spots = len(order_positions)
        amp = 1.0 / np.sqrt(n_spots) if n_spots > 0 else 1.0

        for py, px in order_positions:
            if 0 <= py < tot_orders_y and 0 <= px < tot_orders_x:
                pattern[0, :, py, px] = amp

        return self.normalize(pattern)

    def _generate_asm_target(self, params: dict) -> torch.Tensor:
        """Generate target for ASM optimization (Strategy 1).

        Creates a full-resolution pattern with spots at physical positions.
        The output plane size is approximately equal to the input plane size
        for ASM propagation.
        """
        # Get full resolution (same as DOE size)
        resolution = self.config.phase_resolution
        h, w = resolution

        # Get physical parameters
        pixel_size = self.config.device.pixel_size
        z = params.get('working_distance')

        # Create empty pattern
        pattern = torch.zeros(
            1, self.num_channels, h, w,
            device=self.device
        )

        # Get target positions (physical, in meters)
        target_positions = params.get('target_positions', [])
        if not target_positions:
            # Fall back to order_positions if no physical positions
            return self._generate_periodic_target(params)

        # Physical size of output plane (same as input for ASM)
        output_size_y = h * pixel_size
        output_size_x = w * pixel_size

        # Amplitude per spot
        n_spots = len(target_positions)
        amp = 1.0 / np.sqrt(n_spots) if n_spots > 0 else 1.0

        # Spot size in pixels (use ~3 pixels radius for each spot)
        spot_radius = 3

        # Place spots at physical positions
        for pos_y, pos_x in target_positions:
            # Convert physical position to pixel (centered coordinate system)
            py = int(h / 2 + pos_y / pixel_size)
            px = int(w / 2 + pos_x / pixel_size)

            # Check bounds
            if 0 <= py < h and 0 <= px < w:
                # Create a small circular spot
                for dy in range(-spot_radius, spot_radius + 1):
                    for dx in range(-spot_radius, spot_radius + 1):
                        if dy * dy + dx * dx <= spot_radius * spot_radius:
                            y = py + dy
                            x = px + dx
                            if 0 <= y < h and 0 <= x < w:
                                pattern[0, :, y, x] = amp

        return self.normalize(pattern)

    def get_order_info(self) -> dict:
        """Get detailed order information for visualization and analysis.

        Returns dict with:
            - mode: SplitterMode (natural or uniform)
            - period: float (meters)
            - working_orders: List[Tuple[int, int]] (order indices)
            - order_angles: List[Tuple[float, float]] (angles in radians)
            - order_angles_deg: List[Tuple[float, float]] (angles in degrees)
            - order_positions: List[Tuple[int, int]] (pixel positions)
        """
        params = self.config.get_splitter_params()
        if params is None:
            return {}

        # Convert angles to degrees for convenience
        order_angles_deg = [
            (np.degrees(ay), np.degrees(ax))
            for ay, ax in params['order_angles']
        ]

        return {
            'mode': params['mode'],
            'period': params['period'],
            'wavelength': params['wavelength'],
            'working_orders': params['working_orders'],
            'order_angles': params['order_angles'],
            'order_angles_deg': order_angles_deg,
            'order_positions': params['order_positions'],
            'target_span': params['target_span'],
            'target_span_deg': (
                np.degrees(params['target_span'][0]),
                np.degrees(params['target_span'][1])
            ),
        }

    def get_order_positions(self) -> list:
        """Get list of order positions for evaluation.

        Returns:
            List of (row, col) order positions in target pattern
        """
        params = self.config.get_splitter_params()
        if params is None:
            return []
        return params['order_positions']
