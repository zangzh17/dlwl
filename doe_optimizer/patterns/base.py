"""Base class for pattern generators."""

from abc import ABC, abstractmethod
from typing import Tuple
import torch
import numpy as np

from ..core.config import DOEConfig, DOEType


class PatternGenerator(ABC):
    """Base class for target pattern generation."""

    def __init__(self, config: DOEConfig, device: torch.device = None):
        """Initialize pattern generator.

        Args:
            config: DOE configuration
            device: Torch device
        """
        self.config = config
        self.device = device or torch.device('cpu')

    @abstractmethod
    def generate(self) -> torch.Tensor:
        """Generate target amplitude pattern.

        Returns:
            Target amplitude tensor [1, C, H, W] normalized
        """
        pass

    @property
    def output_resolution(self) -> Tuple[int, int]:
        """Get output pattern resolution."""
        if self.config.target.roi_resolution:
            return self.config.target.roi_resolution
        return self.config.phase_resolution

    @property
    def num_channels(self) -> int:
        """Get number of wavelength channels."""
        return self.config.physical.num_channels

    def normalize(self, pattern: torch.Tensor) -> torch.Tensor:
        """Normalize pattern so sum of intensity = 1.

        Args:
            pattern: Input amplitude pattern

        Returns:
            Normalized amplitude pattern
        """
        energy = torch.sum(pattern ** 2, dim=(-2, -1), keepdim=True)
        return pattern / torch.sqrt(energy + 1e-10)
