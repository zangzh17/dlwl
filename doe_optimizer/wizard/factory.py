"""
Wizard factory for creating appropriate wizard instances.

Provides a unified interface for wizard creation based on DOE type.
"""

from typing import Dict, Any, Union

from .base import BaseWizard, WizardOutput
from .splitter import SplitterWizard
from .diffuser import DiffuserWizard
from .lens import LensWizard
from .custom import CustomPatternWizard


def create_wizard(
    doe_type: str,
    max_resolution: int = 2000
) -> BaseWizard:
    """Create appropriate wizard for a DOE type.

    Args:
        doe_type: DOE type string, one of:
            - 'splitter_1d', 'splitter_2d'
            - 'spot_projector' (same as splitter_2d)
            - 'diffuser'
            - 'lens', 'lens_array'
            - 'deflector' (single-order splitter)
            - 'custom'
        max_resolution: Maximum simulation resolution

    Returns:
        Appropriate wizard instance

    Raises:
        ValueError: If doe_type is not recognized
    """
    doe_type = doe_type.lower()

    if doe_type == 'splitter_1d':
        return SplitterWizard(is_1d=True, max_resolution=max_resolution)

    elif doe_type in ('splitter_2d', 'spot_projector'):
        return SplitterWizard(is_1d=False, max_resolution=max_resolution)

    elif doe_type == 'deflector':
        # Deflector is essentially a single-spot splitter
        return SplitterWizard(is_1d=False, max_resolution=max_resolution)

    elif doe_type == 'diffuser':
        return DiffuserWizard(max_resolution=max_resolution)

    elif doe_type == 'lens':
        return LensWizard(is_array=False, max_resolution=max_resolution)

    elif doe_type == 'lens_array':
        return LensWizard(is_array=True, max_resolution=max_resolution)

    elif doe_type == 'custom':
        return CustomPatternWizard(max_resolution=max_resolution)

    else:
        raise ValueError(f"Unknown DOE type: {doe_type}. "
                         f"Supported types: splitter_1d, splitter_2d, spot_projector, "
                         f"diffuser, lens, lens_array, deflector, custom")


def generate_params(
    user_input: Dict[str, Any],
    max_resolution: int = 2000
) -> WizardOutput:
    """Convenience function to generate parameters from user input.

    This is the main entry point for parameter generation.

    Args:
        user_input: Complete user input dictionary containing:
            - doe_type: DOE type string
            - wavelength: float (meters)
            - working_distance: float or None
            - device_diameter: float (meters)
            - pixel_size: float (meters)
            - target_spec: DOE-specific target parameters
            - optimization: Optimization parameters (optional)
        max_resolution: Maximum simulation resolution

    Returns:
        WizardOutput with all parameters for optimization

    Example:
        user_input = {
            'doe_type': 'splitter_2d',
            'wavelength': 532e-9,
            'device_diameter': 1e-3,
            'pixel_size': 0.5e-6,
            'target_spec': {
                'num_spots': [5, 5],
                'target_type': 'angle',
                'target_span': [0.1, 0.1],
                'grid_mode': 'natural'
            }
        }
        output = generate_params(user_input)
    """
    doe_type = user_input.get('doe_type')
    if doe_type is None:
        raise ValueError("user_input must contain 'doe_type'")

    wizard = create_wizard(doe_type, max_resolution=max_resolution)
    return wizard.generate_params(user_input)


def get_constraints(
    doe_type: str,
    user_input: Dict[str, Any] = None,
    max_resolution: int = 2000
) -> Dict[str, Any]:
    """Get parameter constraints for a DOE type.

    Args:
        doe_type: DOE type string
        user_input: Partial user input for context-dependent constraints
        max_resolution: Maximum simulation resolution

    Returns:
        Dictionary of constraints for frontend validation
    """
    wizard = create_wizard(doe_type, max_resolution=max_resolution)
    return wizard.get_constraints(user_input or {})
