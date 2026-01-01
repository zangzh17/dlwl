"""Diffractive lens pattern generator."""

import math
import numpy as np
import torch

from .base import PatternGenerator
from ..core.config import DOEConfig


class LensGenerator(PatternGenerator):
    """Generate diffractive lens target patterns.

    Creates Airy-like spot patterns at the focal plane.
    Supports normal and cylindrical lenses.
    """

    def generate(self) -> torch.Tensor:
        """Generate lens target pattern (Airy spot).

        Returns:
            Target amplitude tensor [1, C, H, W]
        """
        target = self.config.target
        resolution = self.output_resolution

        focal_length = target.focal_length
        if focal_length is None:
            raise ValueError("focal_length must be specified for lens DOE")

        lens_type = target.lens_type

        # Calculate diffraction-limited spot size
        wavelength = self.config.physical._wavelength_arr.min()
        diameter = self.config.device.diameter

        # Airy disk radius: 1.22 * lambda * f / D
        airy_radius = 1.22 * wavelength * focal_length / diameter

        # Generate Airy pattern
        if lens_type == 'normal':
            pattern = self._generate_airy_2d(resolution, airy_radius)
        elif lens_type == 'cylindrical_x':
            pattern = self._generate_airy_1d(resolution, airy_radius, axis='x')
        elif lens_type == 'cylindrical_y':
            pattern = self._generate_airy_1d(resolution, airy_radius, axis='y')
        else:
            pattern = self._generate_airy_2d(resolution, airy_radius)

        return self.normalize(pattern)

    def _generate_airy_2d(self, resolution: tuple, airy_radius: float) -> torch.Tensor:
        """Generate 2D Airy pattern.

        Args:
            resolution: (H, W) output resolution
            airy_radius: Airy disk radius in meters

        Returns:
            Airy pattern tensor [1, C, H, W]
        """
        h, w = resolution
        feature_size = self.config.get_feature_size(for_phase=True)
        dy, dx = feature_size

        # Coordinate grids
        y = np.linspace(-h/2 * dy, h/2 * dy - dy, h)
        x = np.linspace(-w/2 * dx, w/2 * dx - dx, w)
        X, Y = np.meshgrid(x, y)

        # Radial distance
        R = np.sqrt(X**2 + Y**2)

        # Airy function: (2*J1(x)/x)^2 where x = pi*r/airy_radius
        arg = np.pi * R / airy_radius
        arg[arg == 0] = 1e-10  # Avoid division by zero

        # Use jinc function approximation (Airy pattern is intensity)
        from scipy.special import j1
        airy_intensity = (2 * j1(arg) / arg) ** 2

        # Convert to amplitude
        airy_amp = np.sqrt(airy_intensity)

        pattern = torch.tensor(airy_amp, dtype=torch.float32, device=self.device)
        pattern = pattern.unsqueeze(0).unsqueeze(0).expand(1, self.num_channels, -1, -1)

        return pattern

    def _generate_airy_1d(self, resolution: tuple, airy_radius: float, axis: str) -> torch.Tensor:
        """Generate 1D Airy (sinc) pattern for cylindrical lens.

        Args:
            resolution: (H, W) output resolution
            airy_radius: Airy disk radius in meters
            axis: 'x' or 'y' for cylindrical axis

        Returns:
            1D Airy pattern tensor [1, C, H, W]
        """
        h, w = resolution
        feature_size = self.config.get_feature_size(for_phase=True)
        dy, dx = feature_size

        if axis == 'x':
            # Line focus along x
            y = np.linspace(-h/2 * dy, h/2 * dy - dy, h)
            R = np.abs(y)
        else:
            # Line focus along y
            x = np.linspace(-w/2 * dx, w/2 * dx - dx, w)
            R = np.abs(x)

        arg = np.pi * R / airy_radius
        arg[arg == 0] = 1e-10

        sinc_intensity = (np.sin(arg) / arg) ** 2
        sinc_amp = np.sqrt(sinc_intensity)

        if axis == 'x':
            pattern = torch.tensor(sinc_amp, dtype=torch.float32, device=self.device)
            pattern = pattern.view(1, 1, h, 1).expand(1, self.num_channels, h, w)
        else:
            pattern = torch.tensor(sinc_amp, dtype=torch.float32, device=self.device)
            pattern = pattern.view(1, 1, 1, w).expand(1, self.num_channels, h, w)

        return pattern

    def get_airy_radius(self) -> float:
        """Get theoretical Airy disk radius.

        Returns:
            Airy radius in meters
        """
        wavelength = self.config.physical._wavelength_arr.min()
        diameter = self.config.device.diameter
        focal_length = self.config.target.focal_length
        return 1.22 * wavelength * focal_length / diameter


class LensArrayGenerator(PatternGenerator):
    """Generate microlens array target patterns."""

    def generate(self) -> torch.Tensor:
        """Generate lens array target pattern.

        Returns:
            Target amplitude tensor [1, C, H, W]
        """
        target = self.config.target
        array_size = target.array_size or (3, 3)
        focal_length = target.focal_length

        if focal_length is None:
            raise ValueError("focal_length must be specified for lens array DOE")

        resolution = self.output_resolution
        n_rows, n_cols = array_size

        # Sub-lens aperture
        sub_h = resolution[0] // n_rows
        sub_w = resolution[1] // n_cols

        # Generate single lens pattern
        single_lens = LensGenerator(self.config, self.device)
        # Temporarily modify resolution for sub-lens
        original_roi = self.config.target.roi_resolution
        self.config.target.roi_resolution = (sub_h, sub_w)

        lens_pattern = single_lens._generate_airy_2d(
            (sub_h, sub_w),
            single_lens.get_airy_radius() / n_rows  # Scale for array
        )

        # Restore
        self.config.target.roi_resolution = original_roi

        # Tile to create array
        pattern = lens_pattern.repeat(1, 1, n_rows, n_cols)

        # Crop to exact resolution
        pattern = pattern[:, :, :resolution[0], :resolution[1]]

        return self.normalize(pattern)
