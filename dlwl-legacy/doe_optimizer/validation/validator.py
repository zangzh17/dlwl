"""
Structured parameters validator.

Validates structured parameters against physical and computational constraints.
"""

from typing import Union, TYPE_CHECKING
import numpy as np

from .messages import ValidationResult, ValidationMessage, Severity

if TYPE_CHECKING:
    from ..params.base import StructuredParams
    from ..params.fft_params import FFTParams
    from ..params.sfr_params import SFRParams
    from ..params.asm_params import ASMParams


# Default maximum resolution for simulation
DEFAULT_MAX_RESOLUTION = 2000


class StructuredParamsValidator:
    """Validates structured parameters against physical and computational limits.

    Attributes:
        max_resolution: Maximum pixels allowed in each dimension
        max_total_pixels: Maximum total pixels (H * W * upsample^2)

    Example:
        validator = StructuredParamsValidator(max_resolution=2000)
        result = validator.validate(params)
        if not result.is_valid:
            print(result.errors)
    """

    def __init__(
        self,
        max_resolution: int = DEFAULT_MAX_RESOLUTION,
        max_total_pixels: int = None
    ):
        """Initialize validator.

        Args:
            max_resolution: Maximum pixels in each dimension
            max_total_pixels: Maximum total simulation pixels (default: max_resolution^2)
        """
        self.max_resolution = max_resolution
        self.max_total_pixels = max_total_pixels or (max_resolution ** 2)

    def validate(self, params: 'StructuredParams') -> ValidationResult:
        """Validate structured parameters.

        Performs the following checks:
        1. Resolution limits
        2. Sampling theorem (Nyquist criterion)
        3. Physical constraints (diffraction angles, etc.)
        4. Parameter consistency

        Args:
            params: Structured parameters to validate

        Returns:
            ValidationResult with errors and warnings
        """
        from ..params.fft_params import FFTParams
        from ..params.sfr_params import SFRParams
        from ..params.asm_params import ASMParams

        result = ValidationResult.success()

        # Type-specific validation
        if isinstance(params, FFTParams):
            self._validate_fft_params(params, result)
        elif isinstance(params, SFRParams):
            self._validate_sfr_params(params, result)
        elif isinstance(params, ASMParams):
            self._validate_asm_params(params, result)
        else:
            result.add_error(
                code="UNKNOWN_PARAM_TYPE",
                message=f"Unknown parameter type: {type(params).__name__}"
            )

        return result

    def _validate_fft_params(self, params: 'FFTParams', result: ValidationResult) -> None:
        """Validate FFT (Type A) parameters."""
        # Check resolution limits
        h, w = params.period_pixels
        up = params.upsample_factor
        total = h * w * (up ** 2)

        if h * up > self.max_resolution or w * up > self.max_resolution:
            result.add_error(
                code="RESOLUTION_EXCEEDED",
                message=f"Simulation resolution {h*up}x{w*up} exceeds limit {self.max_resolution}",
                field="upsample_factor",
                suggestion=f"Reduce upsample_factor to {self.max_resolution // max(h, w)}",
                details={'current': h * up, 'max': self.max_resolution}
            )

        if total > self.max_total_pixels:
            result.add_error(
                code="TOTAL_PIXELS_EXCEEDED",
                message=f"Total simulation pixels {total:,} exceeds limit {self.max_total_pixels:,}",
                field="upsample_factor"
            )

        # Check sampling theorem
        # For FFT, the maximum diffraction angle is limited by the pixel size
        # theta_max = arcsin(lambda / (2 * pixel_size))
        wl = params.physical.wavelength
        ps = params.physical.pixel_size / up  # Effective pixel size after upsampling

        # Check if pixel size is too large for the wavelength
        if ps > wl / 2:
            result.add_warning(
                code="ALIASING_RISK",
                message=f"Pixel size {ps*1e6:.2f}um may cause aliasing (limit: {wl/2*1e6:.2f}um)",
                field="pixel_size",
                suggestion="Increase upsample_factor or reduce pixel_size"
            )

        # Check period size for splitter applications
        dh, dw = params.doe_total_pixels
        nh, nw = dh // h, dw // w
        if nh < 2 or nw < 2:
            result.add_warning(
                code="FEW_PERIODS",
                message=f"Only {nh}x{nw} periods fit in device - efficiency may be reduced",
                field="period_pixels"
            )

    def _validate_sfr_params(self, params: 'SFRParams', result: ValidationResult) -> None:
        """Validate SFR (Type B) parameters."""
        # Check DOE resolution
        h, w = params.doe_pixels
        up = params.upsample_factor

        if h * up > self.max_resolution or w * up > self.max_resolution:
            result.add_error(
                code="RESOLUTION_EXCEEDED",
                message=f"DOE simulation resolution {h*up}x{w*up} exceeds limit {self.max_resolution}",
                field="upsample_factor"
            )

        # Check target resolution
        th, tw = params.target_resolution
        if th > self.max_resolution or tw > self.max_resolution:
            result.add_error(
                code="TARGET_RESOLUTION_EXCEEDED",
                message=f"Target resolution {th}x{tw} exceeds limit {self.max_resolution}",
                field="target_resolution"
            )

        # Check zoom factors are reasonable
        zfy, zfx = params.zoom_factors
        if zfy > 100 or zfx > 100:
            result.add_warning(
                code="LARGE_ZOOM",
                message=f"Large zoom factors ({zfy:.1f}x, {zfx:.1f}x) may reduce accuracy",
                field="target_size",
                suggestion="Consider using a larger DOE or smaller target"
            )

        # Check sampling in output plane
        wl = params.physical.wavelength
        for i, dist in enumerate(params.working_distances):
            # Minimum resolvable feature at target
            min_feature = wl * dist / (h * params.physical.pixel_size)
            target_pixel = params.target_pixel_size[0]

            if target_pixel > min_feature:
                result.add_warning(
                    code="TARGET_UNDERSAMPLED",
                    message=f"Target plane may be undersampled at distance {dist*1e3:.1f}mm",
                    field="target_resolution"
                )

    def _validate_asm_params(self, params: 'ASMParams', result: ValidationResult) -> None:
        """Validate ASM (Type C) parameters."""
        # Check DOE resolution
        dh, dw = params.doe_pixels
        th, tw = params.target_pixels
        up = params.upsample_factor

        if th * up > self.max_resolution or tw * up > self.max_resolution:
            result.add_error(
                code="RESOLUTION_EXCEEDED",
                message=f"Simulation resolution {th*up}x{tw*up} exceeds limit {self.max_resolution}",
                field="upsample_factor"
            )

        # Check total pixels with padding
        total = th * tw * (up ** 2)
        if total > self.max_total_pixels:
            result.add_error(
                code="TOTAL_PIXELS_EXCEEDED",
                message=f"Total pixels with padding {total:,} exceeds limit {self.max_total_pixels:,}",
                details={'target_pixels': (th, tw), 'upsample': up}
            )

        # Check working distance is appropriate for ASM
        wl = params.physical.wavelength
        ps = params.physical.pixel_size
        doe_size = max(dh, dw) * ps

        for dist in params.working_distances:
            # Fresnel number: N = a^2 / (lambda * z)
            # ASM is accurate when N >> 1 (near field)
            fresnel_num = (doe_size ** 2) / (wl * dist)
            if fresnel_num < 1:
                result.add_warning(
                    code="FAR_FIELD_REGIME",
                    message=f"Working distance {dist*1e3:.1f}mm may be in far-field regime (Fresnel N={fresnel_num:.2f})",
                    suggestion="Consider using SFR propagation for far-field"
                )

        # Check if target is too different from DOE
        if th > 3 * dh or tw > 3 * dw:
            result.add_warning(
                code="LARGE_PADDING",
                message=f"Target is much larger than DOE ({th/dh:.1f}x, {tw/dw:.1f}x)",
                suggestion="Consider using SFR propagation for efficiency"
            )


def validate_params(
    params: 'StructuredParams',
    max_resolution: int = DEFAULT_MAX_RESOLUTION
) -> ValidationResult:
    """Convenience function to validate parameters.

    Args:
        params: Structured parameters to validate
        max_resolution: Maximum resolution limit

    Returns:
        ValidationResult
    """
    validator = StructuredParamsValidator(max_resolution=max_resolution)
    return validator.validate(params)
