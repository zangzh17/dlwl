"""
Custom pattern wizard for user-provided target images.

Handles:
- Image loading and preprocessing
- Resolution and period estimation
- Multiple propagation strategies
"""

import math
from typing import Dict, Any, Optional, Union
from pathlib import Path
import torch
import numpy as np

from .base import BaseWizard, WizardOutput
from ..params.base import PhysicalConstants, PropagationType
from ..params.fft_params import FFTParams
from ..params.sfr_params import SFRParams


class CustomPatternWizard(BaseWizard):
    """Wizard for custom pattern DOE parameter generation.

    Supports loading target patterns from:
    - File path
    - Base64 encoded image
    - NumPy array
    - PIL Image
    """

    def generate_params(
        self,
        user_input: Dict[str, Any],
        device: torch.device = None
    ) -> WizardOutput:
        """Generate custom pattern parameters from user input.

        Args:
            user_input: User input containing:
                - wavelength: float (meters)
                - working_distance: float or None
                - device_diameter: float (meters)
                - pixel_size: float (meters)
                - target_spec: dict with:
                    - image_data: str (path or base64) or ndarray
                    - target_resolution: [H, W] (optional)
                    - target_type: 'angle' or 'size'
                    - target_span: float
                    - tolerance: float (for period estimation)
            device: Torch device

        Returns:
            WizardOutput with custom pattern parameters
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
        image_data = target_spec['image_data']
        target_resolution = target_spec.get('target_resolution')
        target_type = target_spec.get('target_type', 'angle')
        target_span = target_spec.get('target_span', 0.1)
        tolerance = target_spec.get('tolerance', 0.05)

        if isinstance(target_span, (list, tuple)):
            target_span = target_span[0]

        # Create physical constants
        physical = PhysicalConstants(
            wavelength=wavelength,
            refraction_index=refraction_index,
            pixel_size=pixel_size
        )

        device_pixels = int(round(device_diameter / pixel_size))
        warnings = []
        self._validate_sampling(wavelength, pixel_size, warnings)

        # Load and preprocess image
        target_image = self._load_image(image_data)

        # Determine target resolution
        if target_resolution is None:
            target_resolution = list(target_image.shape[:2])

        # Resize if needed
        if tuple(target_resolution) != target_image.shape[:2]:
            target_image = self._resize_image(target_image, target_resolution)

        # Convert to angle span if size-based
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

        # Estimate period based on tolerance
        period = self._estimate_period(
            wavelength=wavelength,
            theta_span=theta_span,
            tolerance=tolerance,
            device_diameter=device_diameter
        )
        period_pixels = int(round(period / pixel_size))

        # Choose propagation type
        if working_distance is None:
            # Infinite distance: FFT
            # Use period-sized or full device optimization
            opt_resolution = min(period_pixels, device_pixels)
            structured_params = FFTParams(
                period_pixels=(opt_resolution, opt_resolution),
                doe_total_pixels=(device_pixels, device_pixels),
                physical=physical,
                upsample_factor=1,
                aperture_type='square'
            )
            prop_type = PropagationType.FFT
        else:
            # Finite distance: SFR
            structured_params = SFRParams(
                doe_pixels=(device_pixels, device_pixels),
                physical=physical,
                working_distances=[working_distance],
                target_size=(target_size, target_size),
                target_resolution=tuple(target_resolution),
                upsample_factor=1,
                aperture_type='square'
            )
            prop_type = PropagationType.SFR

        propagator_config = structured_params.to_propagator_config()
        optimization_config = self._create_optimization_config(user_input)

        # Convert image to target pattern tensor
        target_pattern = self._image_to_target(target_image, structured_params)

        computed_values = {
            'period': period,
            'period_pixels': period_pixels,
            'theta_span': theta_span,
            'target_resolution': target_resolution,
            'image_shape': list(target_image.shape),
        }

        return WizardOutput(
            structured_params=structured_params,
            propagator_config=propagator_config,
            optimization_config=optimization_config,
            target_pattern=target_pattern,
            computed_values=computed_values,
            warnings=warnings,
            metadata={'target_type': target_type}
        )

    def get_constraints(self, user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Get constraints for frontend."""
        wavelength = user_input.get('wavelength', 532e-9)
        pixel_size = user_input.get('pixel_size', 0.5e-6)

        max_sin = wavelength / (2 * pixel_size)
        max_angle = math.asin(min(1, max_sin)) if max_sin < 1 else math.pi / 2

        return {
            'target_resolution': {
                'min': 32,
                'max': self.max_resolution,
            },
            'target_span': {
                'min_angle': 0.001,
                'max_angle': 2 * max_angle,
            },
            'tolerance': {
                'min': 0.001,
                'max': 0.5,
                'default': 0.05,
            },
        }

    def _load_image(self, image_data: Union[str, np.ndarray]) -> np.ndarray:
        """Load image from various sources.

        Args:
            image_data: File path, base64 string, or numpy array

        Returns:
            Grayscale image as numpy array [H, W]
        """
        if isinstance(image_data, np.ndarray):
            image = image_data
        elif isinstance(image_data, str):
            if Path(image_data).exists():
                # Load from file
                try:
                    from PIL import Image
                    img = Image.open(image_data)
                    image = np.array(img)
                except ImportError:
                    import cv2
                    image = cv2.imread(image_data)
            else:
                # Assume base64
                import base64
                from io import BytesIO
                try:
                    from PIL import Image
                    img_bytes = base64.b64decode(image_data)
                    img = Image.open(BytesIO(img_bytes))
                    image = np.array(img)
                except Exception as e:
                    raise ValueError(f"Could not decode image data: {e}")
        else:
            raise ValueError(f"Unsupported image_data type: {type(image_data)}")

        # Convert to grayscale if needed
        if len(image.shape) == 3:
            if image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]  # Drop alpha
            # Luminance conversion
            image = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]

        # Normalize to [0, 1]
        image = image.astype(np.float32)
        if image.max() > 1:
            image = image / 255.0

        # Apply inverse gamma (sRGB to linear)
        image = np.where(image <= 0.04045,
                         image / 12.92,
                         ((image + 0.055) / 1.055) ** 2.4)

        return image

    def _resize_image(self, image: np.ndarray, target_size: list) -> np.ndarray:
        """Resize image to target resolution."""
        try:
            from PIL import Image
            img = Image.fromarray((image * 255).astype(np.uint8))
            img = img.resize((target_size[1], target_size[0]), Image.LANCZOS)
            return np.array(img).astype(np.float32) / 255.0
        except ImportError:
            import cv2
            resized = cv2.resize(image, (target_size[1], target_size[0]),
                                 interpolation=cv2.INTER_LANCZOS4)
            return resized

    def _estimate_period(
        self,
        wavelength: float,
        theta_span: float,
        tolerance: float,
        device_diameter: float
    ) -> float:
        """Estimate optimal period for custom pattern."""
        # Similar to uniform grid splitter
        sin_theta_max = math.sin(theta_span / 2)
        delta_sin_theta = 2 * sin_theta_max

        if delta_sin_theta > 0 and tolerance > 0:
            period = wavelength / (2 * tolerance * delta_sin_theta)
        else:
            period = device_diameter

        return min(period, device_diameter)

    def _image_to_target(
        self,
        image: np.ndarray,
        structured_params
    ) -> torch.Tensor:
        """Convert image to target amplitude pattern."""
        # Get expected size from params
        if hasattr(structured_params, 'period_pixels'):
            target_size = structured_params.period_pixels
        else:
            target_size = structured_params.target_resolution

        # Resize if needed
        if image.shape[:2] != tuple(target_size):
            image = self._resize_image(image, list(target_size))

        # Convert intensity to amplitude
        amplitude = np.sqrt(np.maximum(image, 0))

        # Convert to tensor
        target = torch.from_numpy(amplitude).to(device=self.device, dtype=self.dtype)
        target = target.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        # Normalize
        target = target / (target.sum() + 1e-10).sqrt()

        return target
