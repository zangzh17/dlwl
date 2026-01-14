"""
Lens wizard for diffractive lens and lens array DOEs.

Handles:
- Single lens and lens array configurations
- Normal and cylindrical lens types
- Focal efficiency optimization
"""

import math
from typing import Dict, Any, Tuple, Optional
import torch
import numpy as np

from .base import BaseWizard, WizardOutput
from ..params.base import PhysicalConstants, PropagationType
from ..params.sfr_params import SFRParams
from ..params.asm_params import ASMParams
from ..params.optimization import LossType, LossConfig, OptimizationConfig


class LensWizard(BaseWizard):
    """Wizard for diffractive lens parameter generation.

    Supports:
    - Normal 2D lenses (Airy pattern focus)
    - Cylindrical lenses (1D focusing)
    - Lens arrays
    """

    def __init__(self, is_array: bool = False, max_resolution: int = 2000):
        """Initialize lens wizard.

        Args:
            is_array: True for lens array, False for single lens
            max_resolution: Maximum simulation resolution
        """
        super().__init__(max_resolution)
        self.is_array = is_array

    def generate_params(
        self,
        user_input: Dict[str, Any],
        device: torch.device = None
    ) -> WizardOutput:
        """Generate lens parameters from user input.

        Args:
            user_input: User input containing:
                - wavelength: float (meters)
                - device_diameter: float (meters)
                - pixel_size: float (meters)
                - target_spec: dict with:
                    - focal_length: float (meters)
                    - lens_type: 'normal', 'cylindrical_x', 'cylindrical_y'
                    - array_size: [ny, nx] (for lens arrays)
            device: Torch device

        Returns:
            WizardOutput with lens parameters
        """
        if device is not None:
            self.device = device

        # Extract parameters
        wavelength = user_input['wavelength']
        device_diameter = user_input['device_diameter']
        pixel_size = user_input['pixel_size']
        refraction_index = user_input.get('refraction_index', 1.62)

        target_spec = user_input['target_spec']
        focal_length = target_spec['focal_length']
        lens_type = target_spec.get('lens_type', 'normal')
        array_size = target_spec.get('array_size')

        if self.is_array and array_size is None:
            array_size = (3, 3)  # Default 3x3 array
        elif not self.is_array:
            array_size = (1, 1)

        # Create physical constants
        physical = PhysicalConstants(
            wavelength=wavelength,
            refraction_index=refraction_index,
            pixel_size=pixel_size
        )

        device_pixels = int(round(device_diameter / pixel_size))
        warnings = []
        self._validate_sampling(wavelength, pixel_size, warnings)

        # Compute Airy disk radius
        if self.is_array:
            lens_diameter = device_diameter / max(array_size)
        else:
            lens_diameter = device_diameter

        airy_radius = 1.22 * wavelength * focal_length / lens_diameter

        # Target plane size (at least 2x Airy radius for each spot)
        if self.is_array:
            target_size = device_diameter  # Same as DOE for array
        else:
            target_size = 6 * airy_radius  # 6x Airy radius for single lens

        # For lens, target is small (a few Airy radii), so use ASM
        # Compute target pixels based on target size and pixel size
        target_pixels = max(
            int(target_size / pixel_size),
            device_pixels
        )
        target_pixels = min(target_pixels, self.max_resolution)

        # Create structured parameters (use ASM for small target focusing)
        structured_params = ASMParams(
            doe_pixels=(device_pixels, device_pixels),
            physical=physical,
            working_distances=[focal_length],
            target_pixels=(target_pixels, target_pixels),
            upsample_factor=1,
            aperture_type='circular'
        )

        propagator_config = structured_params.to_propagator_config()

        # Target resolution for pattern generation
        target_resolution = target_pixels

        # Create optimization config with focal efficiency loss
        optimization_config = self._create_lens_optimization_config(
            user_input=user_input,
            airy_radius_pixels=airy_radius / (target_size / target_resolution),
            array_size=array_size if self.is_array else None
        )

        # Generate target pattern
        target_pattern = self._generate_target(
            target_resolution=target_resolution,
            lens_type=lens_type,
            array_size=array_size,
            airy_radius_pixels=airy_radius / (target_size / target_resolution)
        )

        # Compute airy radius in pixels
        target_pixel_size = target_size / target_resolution if target_resolution > 0 else pixel_size
        airy_radius_pixels = airy_radius / target_pixel_size

        # Count non-zero pixels in target pattern
        num_orders = int((target_pattern > 0).sum().item())

        computed_values = {
            'focal_length': focal_length,
            'airy_radius': airy_radius,
            'airy_radius_pixels': airy_radius_pixels,
            'lens_diameter': lens_diameter,
            'array_size': array_size,
            'target_size': target_size,
            'target_pixels': target_pixels,
            'num_orders': num_orders,
        }

        return WizardOutput(
            structured_params=structured_params,
            propagator_config=propagator_config,
            optimization_config=optimization_config,
            target_pattern=target_pattern,
            computed_values=computed_values,
            warnings=warnings,
            metadata={
                'is_array': self.is_array,
                'lens_type': lens_type,
            }
        )

    def get_constraints(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Get constraints for frontend."""
        device_diameter = user_input.get('device_diameter', 1e-3)

        return {
            'focal_length': {
                'min': device_diameter * 0.1,  # At least f/0.1
                'max': device_diameter * 1000,  # Up to f/1000
            },
            'lens_type': ['normal', 'cylindrical_x', 'cylindrical_y'],
            'array_size': {
                'min': 1,
                'max': 20,
            },
        }

    def _create_lens_optimization_config(
        self,
        user_input: Dict[str, Any],
        airy_radius_pixels: float,
        array_size: Optional[Tuple[int, int]]
    ) -> OptimizationConfig:
        """Create optimization config for lens with focal efficiency loss."""
        from ..params.optimization import OptMethod

        opt_input = user_input.get('optimization', {})

        # Use focal efficiency loss for lenses
        loss_config = LossConfig(
            loss_type=LossType.FOCAL_EFFICIENCY,
            focal_params={
                'airy_radius': airy_radius_pixels,
                'array_size': array_size,
            }
        )

        phase_method_str = opt_input.get('phase_method', 'SGD')
        try:
            phase_method = OptMethod(phase_method_str)
        except ValueError:
            phase_method = OptMethod.SGD

        return OptimizationConfig(
            phase_method=phase_method,
            phase_lr=opt_input.get('phase_lr', 1e-8),
            phase_iters=opt_input.get('phase_iters', 15000),
            loss=loss_config,
            fab_enabled=opt_input.get('fab_enabled', False),
        )

    def _generate_target(
        self,
        target_resolution: int,
        lens_type: str,
        array_size: Tuple[int, int],
        airy_radius_pixels: float
    ) -> torch.Tensor:
        """Generate Airy disk target pattern."""
        h = w = target_resolution
        target = torch.zeros(1, 1, h, w, device=self.device, dtype=self.dtype)

        ny, nx = array_size
        spot_spacing_y = h // ny if ny > 1 else 0
        spot_spacing_x = w // nx if nx > 1 else 0

        for iy in range(ny):
            for ix in range(nx):
                if ny == 1 and nx == 1:
                    cy, cx = h // 2, w // 2
                else:
                    cy = h // 2 + (iy - ny // 2) * spot_spacing_y
                    cx = w // 2 + (ix - nx // 2) * spot_spacing_x

                y_coords = torch.arange(h, device=self.device, dtype=self.dtype) - cy
                x_coords = torch.arange(w, device=self.device, dtype=self.dtype) - cx
                yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

                if lens_type == 'cylindrical_x':
                    # Focus only in x direction
                    r = torch.abs(xx) / airy_radius_pixels
                elif lens_type == 'cylindrical_y':
                    # Focus only in y direction
                    r = torch.abs(yy) / airy_radius_pixels
                else:
                    # Normal 2D Airy pattern
                    r = torch.sqrt(yy**2 + xx**2) / airy_radius_pixels

                # Airy pattern: (2 * J1(pi*r) / (pi*r))^2
                # Approximate with Gaussian for simplicity
                airy = torch.exp(-r**2 / 2)
                target[0, 0] += airy

        # Normalize
        target = target / (target.sum() + 1e-10).sqrt()

        return target
