"""Spot projector pattern generator."""

import math
import numpy as np
import torch

from .base import PatternGenerator
from ..core.config import DOEConfig


class SpotProjectorGenerator(PatternGenerator):
    """Generate 2D spot projector patterns.

    Similar to 2D splitter but with Gaussian spots instead of delta functions.
    """

    def generate(self) -> torch.Tensor:
        """Generate spot projector target pattern.

        Returns:
            Target amplitude tensor [1, C, H, W]
        """
        target = self.config.target
        resolution = self.output_resolution

        num_spots = target.num_spots or (5, 5)
        n_rows, n_cols = num_spots

        h, w = resolution
        feature_size = self.config.get_feature_size(for_phase=True)

        # Calculate spot FWHM (diffraction-limited)
        wavelength = self.config.physical._wavelength_arr.min()
        diameter = self.config.device.diameter

        if self.config.physical.working_distance is not None:
            # Finite distance
            z = self.config.physical._working_distance_arr.min()
            spot_fwhm = 1.22 * wavelength * z / diameter
        else:
            # Infinite distance (angle space)
            # FWHM in angle: ~ lambda / D
            angle_fwhm = wavelength / diameter
            # Convert to pixels at resolution
            total_angle = max(target.target_span)
            spot_fwhm = angle_fwhm / total_angle * max(h, w) * feature_size[0]

        # Generate Gaussian spots at grid positions
        pattern = torch.zeros(1, self.num_channels, h, w, device=self.device)

        # Grid spacing
        spacing_h = h / (n_rows + 1)
        spacing_w = w / (n_cols + 1)

        # Sigma from FWHM
        sigma = spot_fwhm / (2.355 * feature_size[0])  # in pixels

        y = torch.arange(h, device=self.device, dtype=torch.float32)
        x = torch.arange(w, device=self.device, dtype=torch.float32)
        Y, X = torch.meshgrid(y, x, indexing='ij')

        for i in range(n_rows):
            for j in range(n_cols):
                center_y = spacing_h * (i + 1)
                center_x = spacing_w * (j + 1)

                # Gaussian spot
                spot = torch.exp(-((Y - center_y)**2 + (X - center_x)**2) / (2 * sigma**2))
                pattern[0, :, :, :] += spot

        return self.normalize(pattern)

    def get_spot_positions(self) -> list:
        """Get list of spot center positions.

        Returns:
            List of (row, col) pixel positions
        """
        num_spots = self.config.target.num_spots or (5, 5)
        n_rows, n_cols = num_spots
        resolution = self.output_resolution
        h, w = resolution

        spacing_h = h / (n_rows + 1)
        spacing_w = w / (n_cols + 1)

        positions = []
        for i in range(n_rows):
            for j in range(n_cols):
                center_y = int(spacing_h * (i + 1))
                center_x = int(spacing_w * (j + 1))
                positions.append((center_y, center_x))

        return positions
