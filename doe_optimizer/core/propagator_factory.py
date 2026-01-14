"""
Propagator factory for creating propagation functions.

This module separates propagator creation from optimizer creation,
allowing for independent testing and more flexible configuration.
"""

from typing import Callable, Optional, Tuple, Union
from functools import partial
import numpy as np
import torch

from ..params.base import PropagatorConfig, PropagationType
from .propagation import propagation_ASM, propagation_FFT, propagation_SFR


class PropagatorBuilder:
    """Builder for creating propagation functions from configuration.

    Encapsulates propagator creation logic and provides:
    - Automatic kernel precomputation
    - Consistent interface across propagation types
    - Support for multi-channel (multi-wavelength/distance) propagation
    """

    def __init__(
        self,
        config: PropagatorConfig,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32
    ):
        """Initialize propagator builder.

        Args:
            config: Propagator configuration
            device: Torch device
            dtype: Torch dtype
        """
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype

        # Cached kernels
        self._precomputed_H = None
        self._zfft2 = None

    def build(self) -> Callable:
        """Build propagation function.

        Returns:
            Callable propagation function with signature:
            propagator(field: torch.Tensor) -> torch.Tensor

            Input field: [B, 1, H, W] complex tensor
            Output field: [B, C, H_out, W_out] complex tensor
        """
        if self.config.prop_type == PropagationType.FFT:
            return self._build_fft_propagator()
        elif self.config.prop_type == PropagationType.ASM:
            return self._build_asm_propagator()
        elif self.config.prop_type == PropagationType.SFR:
            return self._build_sfr_propagator()
        else:
            raise ValueError(f"Unknown propagation type: {self.config.prop_type}")

    def _build_fft_propagator(self) -> Callable:
        """Build FFT propagator for k-space / infinite distance."""
        output_resolution = self.config.output_resolution

        def propagator(field: torch.Tensor) -> torch.Tensor:
            """FFT propagation to k-space.

            Args:
                field: Input complex field [B, 1, H, W]

            Returns:
                Output field in k-space [B, C, H, W]
            """
            return propagation_FFT(
                field,
                output_resolution=output_resolution,
                z=1  # Positive z for forward propagation
            )

        return propagator

    def _build_asm_propagator(self) -> Callable:
        """Build ASM propagator for near-field propagation with configurable output size."""
        feature_size = self.config.feature_size
        wavelength = self.config.wavelength_array
        prop_dist = self.config.working_distance_array
        output_size = self.config.output_size  # NEW: physical output size
        output_resolution = self.config.output_resolution

        def propagator(
            field: torch.Tensor,
            precomputed_H: torch.Tensor = None
        ) -> torch.Tensor:
            """ASM propagation with configurable output size.

            Args:
                field: Input complex field [B, 1, H, W]
                precomputed_H: Optional precomputed transfer function

            Returns:
                Output field [B, C, H_out, W_out]
            """
            return propagation_ASM(
                field,
                feature_size=feature_size,
                wavelength=wavelength,
                z=prop_dist,
                output_size=output_size,  # NEW: pass output_size
                output_resolution=output_resolution,
                precomputed_H=precomputed_H,
                dtype=self.dtype
            )

        return propagator

    def _build_sfr_propagator(self) -> Callable:
        """Build SFR propagator for large target area propagation."""
        feature_size = self.config.feature_size
        wavelength = self.config.wavelength_array
        prop_dist = self.config.working_distance_array
        output_size = self.config.output_size
        output_resolution = self.config.output_resolution

        def propagator(
            field: torch.Tensor,
            zfft2=None,
            precomputed_H=None
        ) -> torch.Tensor:
            """SFR propagation with zoom-FFT.

            Args:
                field: Input complex field [B, 1, H, W]
                zfft2: Optional ZoomFFT2 instance
                precomputed_H: Optional precomputed kernel

            Returns:
                Output field [B, C, H_out, W_out]
            """
            return propagation_SFR(
                field,
                feature_size=feature_size,
                wavelength=wavelength,
                z=prop_dist,
                output_size=output_size,
                output_resolution=output_resolution,
                zfft2=zfft2,
                precomputed_H=precomputed_H,
                dtype=self.dtype
            )

        return propagator

    def precompute_kernels(self, input_shape: Tuple[int, int]) -> dict:
        """Precompute propagation kernels for efficiency.

        Args:
            input_shape: (H, W) shape of input field

        Returns:
            Dictionary with precomputed kernels
        """
        kernels = {}

        if self.config.prop_type == PropagationType.ASM:
            # Precompute ASM transfer function
            kernels['H'] = self._precompute_asm_kernel(input_shape)

        elif self.config.prop_type == PropagationType.SFR:
            # Precompute SFR kernel and ZoomFFT2
            kernels['H'] = self._precompute_sfr_kernel(input_shape)
            kernels['zfft2'] = self._create_zfft2(input_shape)

        return kernels

    def _precompute_asm_kernel(self, input_shape: Tuple[int, int]) -> torch.Tensor:
        """Precompute ASM transfer function."""
        H, W = input_shape
        dy, dx = self.config.feature_size
        wavelength = self.config.wavelength_array
        prop_dist = self.config.working_distance_array

        # For linear convolution, pad to 2x size
        NH, NW = 2 * H, 2 * W

        # Create frequency grid
        fy = torch.fft.fftfreq(NH, dy, device=self.device, dtype=self.dtype)
        fx = torch.fft.fftfreq(NW, dx, device=self.device, dtype=self.dtype)
        FY, FX = torch.meshgrid(fy, fx, indexing='ij')

        # Convert to tensors
        wl = torch.from_numpy(wavelength).to(device=self.device, dtype=self.dtype)
        z = torch.from_numpy(prop_dist).to(device=self.device, dtype=self.dtype)

        # Compute transfer function for each channel
        # H = exp(j * 2 * pi * z * sqrt(1/lambda^2 - fx^2 - fy^2))
        H_list = []
        for c in range(self.config.num_channels):
            wl_c = wl[0, c, 0, 0] if wl.dim() == 4 else wl[c]
            z_c = z[0, c, 0, 0] if z.dim() == 4 else z[c]

            k = 2 * np.pi / wl_c
            kz_sq = (1 / wl_c) ** 2 - FX ** 2 - FY ** 2
            kz = torch.sqrt(torch.clamp(kz_sq, min=0))

            H_c = torch.exp(1j * 2 * np.pi * z_c * kz)
            # Evanescent wave suppression
            H_c = torch.where(kz_sq > 0, H_c, torch.zeros_like(H_c))

            H_list.append(H_c)

        H = torch.stack(H_list, dim=0).unsqueeze(0)  # [1, C, NH, NW]
        return H

    def _precompute_sfr_kernel(self, input_shape: Tuple[int, int]) -> torch.Tensor:
        """Precompute SFR kernel (Fresnel phase)."""
        H, W = input_shape
        dy, dx = self.config.feature_size
        wavelength = self.config.wavelength_array
        prop_dist = self.config.working_distance_array

        # Create spatial grid
        y = torch.arange(H, device=self.device, dtype=self.dtype) - H / 2
        x = torch.arange(W, device=self.device, dtype=self.dtype) - W / 2
        Y, X = torch.meshgrid(y * dy, x * dx, indexing='ij')

        wl = torch.from_numpy(wavelength).to(device=self.device, dtype=self.dtype)
        z = torch.from_numpy(prop_dist).to(device=self.device, dtype=self.dtype)

        H_list = []
        for c in range(self.config.num_channels):
            wl_c = wl[0, c, 0, 0] if wl.dim() == 4 else wl[c]
            z_c = z[0, c, 0, 0] if z.dim() == 4 else z[c]

            # Fresnel kernel: exp(j * pi * (x^2 + y^2) / (lambda * z))
            phase = np.pi * (X ** 2 + Y ** 2) / (wl_c * z_c)
            H_c = torch.exp(1j * phase)
            H_list.append(H_c)

        H = torch.stack(H_list, dim=0).unsqueeze(0)
        return H

    def _create_zfft2(self, input_shape: Tuple[int, int]):
        """Create ZoomFFT2 instance for SFR."""
        from ..utils.fft_utils import ZoomFFT2

        H, W = input_shape
        out_H, out_W = self.config.output_resolution

        # Calculate zoom factors
        in_size = (H * self.config.feature_size[0], W * self.config.feature_size[1])
        out_size = self.config.output_size

        zoom_y = out_size[0] / in_size[0]
        zoom_x = out_size[1] / in_size[1]

        return ZoomFFT2(
            input_size=(H, W),
            output_size=(out_H, out_W),
            zoom=(zoom_y, zoom_x),
            device=self.device,
            dtype=self.dtype
        )


def build_propagator(
    config: PropagatorConfig,
    device: torch.device = None,
    dtype: torch.dtype = torch.float32
) -> Callable:
    """Convenience function to build a propagator.

    Args:
        config: Propagator configuration
        device: Torch device
        dtype: Torch dtype

    Returns:
        Propagation function
    """
    builder = PropagatorBuilder(config, device=device, dtype=dtype)
    return builder.build()
