"""Diffuser (homogenizer) pattern generator."""

import numpy as np
import torch

from .base import PatternGenerator
from ..core.config import DOEConfig


class DiffuserGenerator(PatternGenerator):
    """Generate diffuser/homogenizer target patterns.

    Creates uniform intensity distribution within specified shape.
    """

    def generate(self) -> torch.Tensor:
        """Generate diffuser target pattern.

        Returns:
            Target amplitude tensor [1, C, H, W]
        """
        target = self.config.target
        resolution = self.output_resolution
        shape = target.diffuser_shape

        h, w = resolution

        if shape == 'circular':
            pattern = self._generate_circular(h, w)
        else:  # square
            pattern = self._generate_square(h, w)

        return self.normalize(pattern)

    def _generate_square(self, h: int, w: int) -> torch.Tensor:
        """Generate square uniform pattern.

        Args:
            h: Height in pixels
            w: Width in pixels

        Returns:
            Square pattern tensor [1, C, H, W]
        """
        # Calculate target size based on target_span
        target_span = self.config.target.target_span
        feature_size = self.config.get_feature_size(for_phase=True)

        if self.config.target.target_type == 'angle':
            # For angle specification, use full ROI
            size_h, size_w = h, w
        else:
            # For size specification, calculate pixels
            size_h = int(target_span[0] / feature_size[0])
            size_w = int(target_span[1] / feature_size[1])
            size_h = min(size_h, h)
            size_w = min(size_w, w)

        pattern = torch.zeros(1, self.num_channels, h, w, device=self.device)

        # Center the square
        start_h = (h - size_h) // 2
        start_w = (w - size_w) // 2

        pattern[:, :, start_h:start_h + size_h, start_w:start_w + size_w] = 1.0

        return pattern

    def _generate_circular(self, h: int, w: int) -> torch.Tensor:
        """Generate circular uniform pattern.

        Args:
            h: Height in pixels
            w: Width in pixels

        Returns:
            Circular pattern tensor [1, C, H, W]
        """
        y = torch.linspace(-1, 1, h, device=self.device)
        x = torch.linspace(-1, 1, w, device=self.device)
        Y, X = torch.meshgrid(y, x, indexing='ij')

        # Create circular mask
        R = torch.sqrt(X**2 + Y**2)
        pattern = (R <= 1.0).float()

        pattern = pattern.unsqueeze(0).unsqueeze(0).expand(1, self.num_channels, -1, -1)

        return pattern
