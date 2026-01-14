"""
Optical propagation models for DOE simulation.

This module provides three propagation methods:
1. ASM (Angular Spectrum Method) - for near-field propagation
2. FFT - for far-field (infinite distance) in angle space
3. SFR (Single Fresnel Transform) - for far-field with adjustable output size

Reference:
- Y. Peng et al., Neural Holography with Camera-in-the-loop Training, SIGGRAPH Asia 2020
"""

import math
import numpy as np
import torch
import torch.fft

from ..utils.fft_utils import ZoomFFT2
from ..utils.image_utils import pad_image, crop_image, fft_interp


def propagation_ASM(
    u_in: torch.Tensor,
    feature_size: tuple,
    wavelength: np.ndarray,
    z: np.ndarray,
    output_size: tuple = None,
    output_resolution: tuple = None,
    linear_conv: bool = True,
    padtype: str = "zero",
    return_H: bool = False,
    precomputed_H: torch.Tensor = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Angular Spectrum Method propagation with configurable output size.

    Args:
        u_in: Input field [B, C, H, W]
        feature_size: (dy, dx) pixel size in meters
        wavelength: Wavelength(s) in meters
        z: Propagation distance(s) in meters
        output_size: (height, width) physical output size in meters. If None, uses input size.
        output_resolution: (height, width) output pixel count. If None, derived from output_size/feature_size.
        linear_conv: Use linear convolution (True) or circular (False)
        padtype: Padding type for linear_conv ("zero" or "median")
        return_H: Return transfer function instead of propagating
        precomputed_H: Pre-computed transfer function
        dtype: Data type for computation

    Returns:
        Propagated field [B, C, H_out, W_out]

    Notes:
        - Case A (output_size > input_size): Zero-pad input to cover output area, then resample
        - Case B (output_size <= input_size): Propagate directly, crop to output area, then resample
        - ASM preserves pixel_size, so physical size is controlled via pixel count
    """
    dy, dx = feature_size
    input_resolution = u_in.size()[-2:]
    input_size = (input_resolution[0] * dy, input_resolution[1] * dx)

    # Default output_size to input_size (original behavior)
    if output_size is None:
        output_size = input_size

    # Calculate simulation size (must cover both input DOE and output target)
    # Case A: output > input -> need to pad input
    # Case B: output <= input -> no padding needed, will crop later
    sim_size_h = max(output_size[0], input_size[0])
    sim_size_w = max(output_size[1], input_size[1])

    # Convert simulation size to pixels (using input pixel_size)
    # Use ceiling to ensure we cover the required physical area
    sim_pixels_h = int(np.ceil(sim_size_h / dy))
    sim_pixels_w = int(np.ceil(sim_size_w / dx))

    # Ensure even pixel count for FFT efficiency (avoid odd-size issues)
    if sim_pixels_h % 2 == 1:
        sim_pixels_h += 1
    if sim_pixels_w % 2 == 1:
        sim_pixels_w += 1

    sim_resolution = (sim_pixels_h, sim_pixels_w)

    # Default output_resolution: match output_size with same pixel_size
    if output_resolution is None:
        out_h = int(np.ceil(output_size[0] / dy))
        out_w = int(np.ceil(output_size[1] / dx))
        output_resolution = (out_h, out_w)

    # Pad input to simulation size if needed (Case A: output > input)
    u_sim = u_in
    if sim_resolution[0] > input_resolution[0] or sim_resolution[1] > input_resolution[1]:
        padval = 0 if padtype == "zero" else torch.median(torch.abs(u_in))
        u_sim = pad_image(u_in, sim_resolution, padval=padval)

    # For linear convolution, further pad to 2x for accurate convolution
    if linear_conv:
        conv_size = [i * 2 for i in sim_resolution]
        padval = 0 if padtype == "zero" else torch.median(torch.abs(u_sim))
        u_sim = pad_image(u_sim, conv_size, padval=padval)

    # Compute transfer function if not provided
    if precomputed_H is None:
        field_resolution = u_sim.size()
        num_y, num_x = field_resolution[2], field_resolution[3]
        y, x = dy * float(num_y), dx * float(num_x)
        zc_x = 2 * num_x * dx**2 / wavelength * np.sqrt(1 - (wavelength / (2 * dx))**2)
        zc_y = 2 * num_y * dy**2 / wavelength * np.sqrt(1 - (wavelength / (2 * dy))**2)
        zc = (zc_x + zc_y) / 2
        sy = np.linspace(-y/2, y/2 - dy, num_y)
        sx = np.linspace(-x/2, x/2 - dx, num_x)
        fy = np.linspace(-1/(2*dy) + 0.5/(2*y), 1/(2*dy) - 0.5/(2*y), num_y)
        fx = np.linspace(-1/(2*dx) + 0.5/(2*x), 1/(2*dx) - 0.5/(2*x), num_x)
        FX, FY = np.meshgrid(fx, fy)
        X, Y = np.meshgrid(sx, sy)
        FX = FX.reshape(1, 1, *FX.shape)
        FY = FY.reshape(1, 1, *FY.shape)
        X = X.reshape(1, 1, *X.shape)
        Y = Y.reshape(1, 1, *Y.shape)
        HH = 2 * math.pi * np.sqrt(np.maximum(1/wavelength**2 - (FX**2 + FY**2), 0))
        fy_max = 1 / np.sqrt((2 * z * (1/y))**2 + 1) / wavelength
        fx_max = 1 / np.sqrt((2 * z * (1/x))**2 + 1) / wavelength
        H_exp = torch.tensor(HH, dtype=dtype).to(u_in.device)
        H_filter = torch.tensor(((np.abs(FX) < fx_max) & (np.abs(FY) < fy_max)).astype(np.uint8), dtype=dtype)
        H_ASM = torch.exp(1j * H_exp * torch.tensor(z, dtype=dtype).to(u_in.device))
        H_ASM = H_ASM * H_filter.to(u_in.device)
        H_ASM = torch.fft.ifftshift(H_ASM)
        R = np.sqrt(X**2 + Y**2 + z**2)
        h = 1/(2*np.pi) * z/R * (1/R - 1j * 2*np.pi/wavelength) * np.exp(1j * R * 2*np.pi/wavelength) / R
        H_RSC = torch.tensor(np.fft.fft2(np.fft.fftshift(h)), dtype=H_ASM.dtype).to(u_in.device)
        H_RSC = H_RSC / ((H_RSC.abs()**2).sum().sqrt() / (H_ASM.abs()**2).sum().sqrt())
        z_filter = z <= zc
        asm_filter = torch.tensor(z_filter.astype(np.uint8), dtype=dtype).to(u_in.device)
        rsc_filter = torch.tensor((~z_filter).astype(np.uint8), dtype=dtype).to(u_in.device)
        H = asm_filter * H_ASM + rsc_filter * H_RSC
    else:
        H = precomputed_H

    if return_H:
        return H

    # Perform ASM propagation
    u_out = H * torch.fft.fftn(torch.fft.ifftshift(u_sim, dim=(-2, -1)), dim=(-2, -1), norm="ortho")
    u_out = torch.fft.fftshift(torch.fft.ifftn(u_out, dim=(-2, -1), norm="ortho"), dim=(-2, -1))

    # Crop back from linear convolution padding
    if linear_conv:
        u_out = crop_image(u_out, sim_resolution, pytorch=True)

    # Crop to output physical size (Case B: output <= sim_size)
    # Calculate output pixels at simulation pixel_size
    output_pixels_h = int(np.ceil(output_size[0] / dy))
    output_pixels_w = int(np.ceil(output_size[1] / dx))

    # Ensure even for consistency
    if output_pixels_h % 2 == 1 and output_pixels_h < sim_resolution[0]:
        output_pixels_h += 1
    if output_pixels_w % 2 == 1 and output_pixels_w < sim_resolution[1]:
        output_pixels_w += 1

    output_pixels = (output_pixels_h, output_pixels_w)

    if output_pixels[0] < sim_resolution[0] or output_pixels[1] < sim_resolution[1]:
        u_out = crop_image(u_out, output_pixels, pytorch=True)

    # Resample to target output_resolution
    current_resolution = u_out.size()[-2:]
    if current_resolution[0] != output_resolution[0] or current_resolution[1] != output_resolution[1]:
        u_out = fft_interp(u_out, output_resolution)

    return u_out


def propagation_FFT(u_in: torch.Tensor, output_resolution: tuple = None, z: float = 1.0) -> torch.Tensor:
    z_arr = np.atleast_1d(z)
    if np.max(z_arr) >= 0:
        u_out = torch.fft.fftshift(torch.fft.fftn(torch.fft.fftshift(u_in, dim=(-2, -1)), dim=(-2, -1), norm="ortho"), dim=(-2, -1))
    else:
        u_out = torch.fft.ifftshift(torch.fft.ifftn(torch.fft.ifftshift(u_in, dim=(-2, -1)), dim=(-2, -1), norm="ortho"), dim=(-2, -1))

    if output_resolution is not None:
        u_out = fft_interp(u_out, output_resolution)

    return u_out


def propagation_SFR(
    u_in: torch.Tensor,
    feature_size: tuple,
    wavelength: np.ndarray,
    z: np.ndarray,
    output_size: tuple = None,
    output_resolution: tuple = None,
    return_zfft2: bool = False,
    zfft2: callable = None,
    return_H: bool = False,
    precomputed_H: tuple = None,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    input_resolution = u_in.size()[-2:]
    ny, nx = input_resolution
    dy, dx = feature_size
    input_size = [ny * dy, nx * dx]

    if output_size is None:
        output_size = input_size
    if output_resolution is None:
        output_resolution = input_resolution

    # Calculate zoom factors
    sy = output_size[0] / (wavelength * z / dy)
    sx = output_size[1] / (wavelength * z / dx)

    # Check if zoom is close to 1.0 and resolutions match - use standard FFT for numerical stability
    # (ZoomFFT has numerical issues with odd-sized arrays)
    zoom_is_unity = np.allclose(sy.squeeze(), 1.0, rtol=1e-6) and np.allclose(sx.squeeze(), 1.0, rtol=1e-6)
    resolution_matches = (input_resolution[0] == output_resolution[0] and
                         input_resolution[1] == output_resolution[1])
    use_standard_fft = zoom_is_unity and resolution_matches

    if zfft2 is None:
        if use_standard_fft:
            # Use standard centered FFT for zoom=1.0 (more numerically stable)
            def zfft2(x):
                return torch.fft.fftshift(
                    torch.fft.fft2(
                        torch.fft.ifftshift(x, dim=(-2, -1)),
                        norm='ortho'
                    ),
                    dim=(-2, -1)
                )
        else:
            ZoomFFT = ZoomFFT2(input_resolution, sy.squeeze(), sx.squeeze(), output_resolution, device=u_in.device, dtype=dtype)
            zfft2 = ZoomFFT.cfft2

    if return_zfft2:
        return zfft2

    if precomputed_H is None:
        num_y, num_x = input_resolution[0], input_resolution[1]
        ys = np.linspace(-input_size[0] / 2, input_size[0] / 2 - input_size[0] / num_y, num_y)
        xs = np.linspace(-input_size[1] / 2, input_size[1] / 2 - input_size[1] / num_x, num_x)
        xs_grid, ys_grid = np.meshgrid(xs, ys)
        xs_grid = xs_grid.reshape(1, 1, *xs_grid.shape)
        ys_grid = ys_grid.reshape(1, 1, *ys_grid.shape)
        H_exp = np.pi / wavelength / z * (xs_grid ** 2 + ys_grid ** 2)
        H_in = torch.tensor(np.exp(1j * H_exp), dtype=torch.complex64).to(u_in.device)

        num_y, num_x = output_resolution[0], output_resolution[1]
        ys = np.linspace(-output_size[0] / 2, output_size[0] / 2 - output_size[0] / num_y, num_y)
        xs = np.linspace(-output_size[1] / 2, output_size[1] / 2 - output_size[1] / num_x, num_x)
        xs_grid, ys_grid = np.meshgrid(xs, ys)
        ys_grid = ys_grid.reshape(1, 1, *ys_grid.shape)
        xs_grid = xs_grid.reshape(1, 1, *xs_grid.shape)
        H_exp = np.pi / wavelength / z * (xs_grid ** 2 + ys_grid ** 2)
        prop_phase = 2 * np.pi * z / wavelength
        H_out = torch.tensor(np.exp(1j * H_exp) / (1j) * np.exp(1j * prop_phase), dtype=torch.complex64).to(u_in.device)
    else:
        H_in, H_out = precomputed_H

    if return_H:
        return H_in, H_out

    # Apply chirped field to FFT
    U = zfft2(H_in * u_in)
    return H_out * U


def get_propagator(prop_model: str):
    propagators = {"ASM": propagation_ASM, "FFT": propagation_FFT, "SFR": propagation_SFR, "None": None}
    return propagators.get(prop_model)
