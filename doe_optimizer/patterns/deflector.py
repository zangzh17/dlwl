"""Deflector (blazed grating) pattern generator."""

import math
import numpy as np
import torch

from .base import PatternGenerator
from ..core.config import DOEConfig


class DeflectorGenerator(PatternGenerator):
    """Generate deflector/blazed grating target patterns.

    Creates a single diffraction order at specified angle.
    """

    def generate(self) -> torch.Tensor:
        """Generate deflector target pattern.

        Returns:
            Target amplitude tensor [1, C, H, W]
        """
        target = self.config.target
        resolution = self.output_resolution

        deflection_angle = target.deflection_angle
        if deflection_angle is None:
            deflection_angle = (0.0, 0.0)

        theta_x, theta_y = deflection_angle

        h, w = resolution
        pattern = torch.zeros(1, self.num_channels, h, w, device=self.device)

        # Calculate order position from angle
        wavelength = self.config.physical._wavelength_arr.min()
        pixel_size = self.config.device.pixel_size

        # Order = sin(theta) * D / lambda where D is total size
        D_y = h * pixel_size
        D_x = w * pixel_size

        order_y = int(round(math.sin(theta_y) * D_y / wavelength))
        order_x = int(round(math.sin(theta_x) * D_x / wavelength))

        # Place delta at order position
        pos_y = h // 2 + order_y
        pos_x = w // 2 + order_x

        if 0 <= pos_y < h and 0 <= pos_x < w:
            pattern[0, :, pos_y, pos_x] = 1.0
        else:
            # If out of range, place at edge
            pos_y = max(0, min(h - 1, pos_y))
            pos_x = max(0, min(w - 1, pos_x))
            pattern[0, :, pos_y, pos_x] = 1.0

        return self.normalize(pattern)

    def get_deflection_order(self) -> tuple:
        """Get the target diffraction order.

        Returns:
            (order_y, order_x) tuple
        """
        target = self.config.target
        deflection_angle = target.deflection_angle or (0.0, 0.0)
        theta_x, theta_y = deflection_angle

        resolution = self.output_resolution
        h, w = resolution

        wavelength = self.config.physical._wavelength_arr.min()
        pixel_size = self.config.device.pixel_size

        D_y = h * pixel_size
        D_x = w * pixel_size

        order_y = int(round(math.sin(theta_y) * D_y / wavelength))
        order_x = int(round(math.sin(theta_x) * D_x / wavelength))

        return (order_y, order_x)
