"""Factory function for creating pattern generators."""

import torch

from ..core.config import DOEConfig, DOEType
from .base import PatternGenerator
from .splitter import SplitterGenerator
from .spot_projector import SpotProjectorGenerator
from .diffuser import DiffuserGenerator
from .lens import LensGenerator, LensArrayGenerator
from .deflector import DeflectorGenerator
from .custom import CustomPatternGenerator


def create_pattern_generator(
    config: DOEConfig,
    device: torch.device = None
) -> PatternGenerator:
    """Create pattern generator based on DOE type.

    Args:
        config: DOE configuration
        device: Torch device

    Returns:
        PatternGenerator instance for the specified DOE type

    Raises:
        ValueError: If DOE type is not supported
    """
    generators = {
        DOEType.SPLITTER_1D: SplitterGenerator,
        DOEType.SPLITTER_2D: SplitterGenerator,
        DOEType.SPOT_PROJECTOR: SpotProjectorGenerator,
        DOEType.DIFFUSER: DiffuserGenerator,
        DOEType.LENS: LensGenerator,
        DOEType.LENS_ARRAY: LensArrayGenerator,
        DOEType.DEFLECTOR: DeflectorGenerator,
        DOEType.CUSTOM: CustomPatternGenerator,
    }

    generator_class = generators.get(config.doe_type)

    if generator_class is None:
        raise ValueError(f"Unsupported DOE type: {config.doe_type}")

    return generator_class(config, device)


def generate_target_pattern(
    config: DOEConfig,
    device: torch.device = None
) -> torch.Tensor:
    """Convenience function to generate target pattern directly.

    Args:
        config: DOE configuration
        device: Torch device

    Returns:
        Target amplitude tensor [1, C, H, W]
    """
    generator = create_pattern_generator(config, device)
    return generator.generate()
