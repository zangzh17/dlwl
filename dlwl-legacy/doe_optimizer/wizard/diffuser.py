"""
Diffuser wizard for uniform illumination patterns.

Handles:
- Square and circular diffusion patterns
- Infinite and finite distance configurations
- SFR propagation for large target areas
"""

import math
from typing import Dict, Any, Tuple
import torch
import numpy as np

from .base import BaseWizard, WizardOutput
from ..params.base import PhysicalConstants, PropagationType
from ..params.fft_params import FFTParams
from ..params.sfr_params import SFRParams


class DiffuserWizard(BaseWizard):
    """Wizard for diffuser/homogenizer parameter generation.

    Diffusers create uniform illumination patterns, either:
    - Square: rectangular uniform pattern
    - Circular: circular uniform pattern (top-hat)
    """

    # Default target margin factor (10% margin)
    DEFAULT_TARGET_MARGIN_FACTOR = 1.1

    def generate_params(
        self,
        user_input: Dict[str, Any],
        device: torch.device = None
    ) -> WizardOutput:
        """Generate diffuser parameters from user input.

        Args:
            user_input: User input containing:
                - wavelength: float (meters)
                - working_distance: float or None
                - device_diameter: float (meters)
                - pixel_size: float (meters)
                - target_spec: dict with:
                    - shape: 'square' or 'circular'
                    - target_type: 'angle' or 'size'
                    - target_span: float (full angle or size)
            device: Torch device

        Returns:
            WizardOutput with diffuser parameters
        """
        if device is not None:
            self.device = device

        # Extract parameters
        wavelength = user_input['wavelength']
        working_distance = user_input.get('working_distance')
        device_diameter = user_input['device_diameter']
        pixel_size = user_input['pixel_size']
        refraction_index = user_input.get('refraction_index', 1.62)

        target_spec = user_input['target_spec']
        shape = target_spec.get('shape', 'square')
        target_type = target_spec.get('target_type', 'angle')
        target_span = target_spec['target_span']
        if isinstance(target_span, (list, tuple)):
            target_span = target_span[0]

        # Get target margin from advanced or optimization settings
        # Frontend sends target_margin as decimal (e.g., 0.1 for 10%)
        advanced_settings = user_input.get('advanced', {})
        opt_settings = user_input.get('optimization', {})
        target_margin = advanced_settings.get('target_margin') or opt_settings.get('target_margin')
        if target_margin is None:
            target_margin = self.DEFAULT_TARGET_MARGIN_FACTOR - 1.0
        target_margin_factor = 1.0 + target_margin

        # Create physical constants
        physical = PhysicalConstants(
            wavelength=wavelength,
            refraction_index=refraction_index,
            pixel_size=pixel_size
        )

        device_pixels = int(round(device_diameter / pixel_size))
        warnings = []
        self._validate_sampling(wavelength, pixel_size, warnings)

        # Convert to angle if size-based
        if target_type == 'size':
            if working_distance is None:
                raise ValueError("Cannot use 'size' target_type with infinite distance")
            theta_span = 2 * math.atan(target_span / (2 * working_distance))
            target_size = target_span
        else:
            theta_span = target_span
            if working_distance:
                target_size = 2 * working_distance * math.tan(theta_span / 2)
            else:
                target_size = None

        # Initialize target_size_with_margin (will be set for SFR)
        target_size_with_margin = target_size

        # Choose propagation type based on target_type
        # - angle: FFT (k-space / angular coordinates)
        # - size: SFR (physical coordinates, requires working_distance)
        if target_type == 'angle':
            # Angle-based: FFT (far field / k-space)
            # Simulation Pixels = DOE Pixels (one period = full device)
            structured_params = self._create_fft_params(
                physical=physical,
                device_pixels=device_pixels,
                theta_span=theta_span
            )
            prop_type = PropagationType.FFT
            target_pattern = self._generate_fft_target(
                device_pixels=device_pixels,
                theta_span=theta_span,
                wavelength=wavelength,
                pixel_size=pixel_size,
                shape=shape
            )
        else:
            # Physical size: SFR (finite distance)
            # Simulation Pixels = DOE Pixels for diffuser
            target_resolution = device_pixels
            # Apply target margin for simulation area
            target_size_with_margin = target_size * target_margin_factor
            structured_params = self._create_sfr_params(
                physical=physical,
                device_pixels=device_pixels,
                working_distance=working_distance,
                target_size=target_size_with_margin,
                target_resolution=target_resolution
            )
            prop_type = PropagationType.SFR
            # Generate target filling proportion based on margin
            # fill_ratio = 1/target_margin_factor to leave margin around
            target_pattern = self._generate_sfr_target(
                target_resolution=target_resolution,
                shape=shape,
                fill_ratio=1.0 / target_margin_factor
            )

        propagator_config = structured_params.to_propagator_config()
        optimization_config = self._create_optimization_config(user_input)

        # Count non-zero pixels in target pattern
        num_orders = int((target_pattern > 0).sum().item())

        # target_size is the user-specified target span (without margin)
        # target_size_with_margin is only used internally for simulation area
        # Preview should show the actual target area (without margin)
        computed_values = {
            'theta_span': theta_span,
            'target_size': target_size,  # Original user-specified size (without margin)
            'target_size_with_margin': target_size_with_margin,  # For internal reference
            'target_margin_factor': target_margin_factor,
            'shape': shape,
            'num_orders': num_orders,
        }

        return WizardOutput(
            structured_params=structured_params,
            propagator_config=propagator_config,
            optimization_config=optimization_config,
            target_pattern=target_pattern,
            computed_values=computed_values,
            warnings=warnings,
            metadata={'shape': shape, 'target_type': target_type}
        )

    def get_constraints(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Get constraints for frontend."""
        wavelength = user_input.get('wavelength', 532e-9)
        pixel_size = user_input.get('pixel_size', 0.5e-6)

        max_sin = wavelength / (2 * pixel_size)
        max_angle = math.asin(min(1, max_sin)) if max_sin < 1 else math.pi / 2

        return {
            'shape': ['square', 'circular'],
            'target_span': {
                'min_angle': 0.001,
                'max_angle': 2 * max_angle,
            },
            'max_angle_degrees': math.degrees(max_angle),
        }

    def _create_fft_params(
        self,
        physical: PhysicalConstants,
        device_pixels: int,
        theta_span: float
    ) -> FFTParams:
        """Create FFT parameters for infinite distance diffuser."""
        return FFTParams(
            period_pixels=(device_pixels, device_pixels),
            doe_total_pixels=(device_pixels, device_pixels),
            physical=physical,
            upsample_factor=1,
            aperture_type='circular'
        )

    def _create_sfr_params(
        self,
        physical: PhysicalConstants,
        device_pixels: int,
        working_distance: float,
        target_size: float,
        target_resolution: int
    ) -> SFRParams:
        """Create SFR parameters for finite distance diffuser."""
        return SFRParams(
            doe_pixels=(device_pixels, device_pixels),
            physical=physical,
            working_distances=[working_distance],
            target_size=(target_size, target_size),
            target_resolution=(target_resolution, target_resolution),
            upsample_factor=1,
            aperture_type='circular'
        )

    def _generate_fft_target(
        self,
        device_pixels: int,
        theta_span: float,
        wavelength: float,
        pixel_size: float,
        shape: str
    ) -> torch.Tensor:
        """Generate FFT target pattern (uniform in angle space)."""
        h = w = device_pixels

        # K-space pixel corresponds to angle: delta_theta = lambda / (N * pixel_size)
        k_step = wavelength / (device_pixels * pixel_size)
        target_k_radius = theta_span / (2 * k_step)  # In k-space pixels

        target = torch.zeros(1, 1, h, w, device=self.device, dtype=self.dtype)

        cy, cx = h // 2, w // 2
        y_coords = torch.arange(h, device=self.device, dtype=self.dtype) - cy
        x_coords = torch.arange(w, device=self.device, dtype=self.dtype) - cx
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        if shape == 'circular':
            mask = (yy**2 + xx**2) <= target_k_radius**2
        else:  # square
            mask = (torch.abs(yy) <= target_k_radius) & (torch.abs(xx) <= target_k_radius)

        target[0, 0, mask] = 1.0

        # Normalize
        target = target / (target.sum() + 1e-10).sqrt()

        return target

    def _generate_sfr_target(
        self,
        target_resolution: int,
        shape: str,
        fill_ratio: float = 1.0
    ) -> torch.Tensor:
        """Generate SFR target pattern (uniform in physical space).

        Args:
            target_resolution: Target pattern resolution
            shape: 'square' or 'circular'
            fill_ratio: Fraction of target to fill (for margin support)
        """
        h = w = target_resolution
        target = torch.zeros(1, 1, h, w, device=self.device, dtype=self.dtype)

        cy, cx = h // 2, w // 2
        # Apply fill_ratio to radius
        base_radius = h // 2 - 1
        radius = int(base_radius * fill_ratio)

        y_coords = torch.arange(h, device=self.device, dtype=self.dtype) - cy
        x_coords = torch.arange(w, device=self.device, dtype=self.dtype) - cx
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')

        if shape == 'circular':
            mask = (yy**2 + xx**2) <= radius**2
        else:  # square
            mask = (torch.abs(yy) <= radius) & (torch.abs(xx) <= radius)

        target[0, 0, mask] = 1.0

        # Normalize
        target = target / (target.sum() + 1e-10).sqrt()

        return target
