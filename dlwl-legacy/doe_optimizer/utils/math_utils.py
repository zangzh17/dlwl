"""Mathematical utility functions."""

import math
import numpy as np
import torch


def height2phase(
    height: torch.Tensor,
    wavelength: np.ndarray,
    refraction_index: np.ndarray,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Convert height profile to phase.

    Args:
        height: Height tensor [B, C, H, W] in meters
        wavelength: Wavelength array [1, C, 1, 1] in meters
        refraction_index: Refractive index array [1, C, 1, 1]
        dtype: Output dtype

    Returns:
        Phase tensor [B, C, H, W] in radians
    """
    wavelength_t = torch.tensor(wavelength, dtype=dtype, device=height.device)
    n_t = torch.tensor(refraction_index, dtype=dtype, device=height.device)

    # phase = 2*pi * height * (n-1) / lambda
    phase = 2 * math.pi * height * (n_t - 1) / wavelength_t

    return phase


def phase2height(
    phase: torch.Tensor,
    wavelength: np.ndarray,
    refraction_index: np.ndarray,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Convert phase to height profile.

    Args:
        phase: Phase tensor [B, C, H, W] in radians
        wavelength: Wavelength array [1, C, 1, 1] in meters
        refraction_index: Refractive index array [1, C, 1, 1]
        dtype: Output dtype

    Returns:
        Height tensor [B, C, H, W] in meters
    """
    wavelength_t = torch.tensor(wavelength, dtype=dtype, device=phase.device)
    n_t = torch.tensor(refraction_index, dtype=dtype, device=phase.device)

    # height = phase * lambda / (2*pi * (n-1))
    height = phase * wavelength_t / (2 * math.pi * (n_t - 1))

    return height


def spherical_phase(
    shape: tuple,
    feature_size: tuple,
    wavelength: np.ndarray,
    focal_length: float,
    dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Generate spherical (lens) phase for converging/diverging beam.

    Args:
        shape: (H, W) or (B, C, H, W) output shape
        feature_size: (dy, dx) pixel size in meters
        wavelength: Wavelength array [1, C, 1, 1] in meters
        focal_length: Focal length in meters (positive=converging)
        dtype: Output dtype

    Returns:
        Spherical phase tensor in radians
    """
    if len(shape) == 4:
        _, _, h, w = shape
    else:
        h, w = shape[-2], shape[-1]

    dy, dx = feature_size
    y = np.linspace(-h/2 * dy, h/2 * dy - dy, h)
    x = np.linspace(-w/2 * dx, w/2 * dx - dx, w)
    X, Y = np.meshgrid(x, y)
    X = X.reshape(1, 1, h, w)
    Y = Y.reshape(1, 1, h, w)

    # Spherical phase: phi = -pi * (x^2 + y^2) / (lambda * f)
    phase = -np.pi * (X**2 + Y**2) / (wavelength * focal_length)

    return torch.tensor(phase, dtype=dtype)


def compute_loss(
    recon: torch.Tensor,
    target: torch.Tensor,
    loss_type: str = 'L2',
    roi_mask: torch.Tensor = None,
    square: bool = True,
    normalize: str = 'sqrt',
) -> torch.Tensor:
    """Compute loss between reconstruction and target.

    Uses L2 norm normalization and RMSE (sqrt of MSE) to match original implementation.

    Args:
        recon: Reconstructed amplitude [B, C, H, W]
        target: Target amplitude [B, C, H, W]
        loss_type: 'L1', 'L2', or 'Laplacian'
        roi_mask: Optional ROI mask [B, C, H, W]
        square: If True, compare intensities (|.|^2) instead of amplitudes
        normalize: Normalization method: 'sqrt' (L2 norm), 'sum', 'max', or 'none'

    Returns:
        Scalar loss value
    """
    if square:
        recon = recon ** 2
        target = target ** 2

    if roi_mask is not None:
        recon = recon * roi_mask
        target = target * roi_mask

    # Normalize using specified method (matching original utils.normalize)
    if normalize == 'sqrt':
        # L2 norm normalization: input / sqrt(sum(input^2)) * sqrt(size)
        sz = recon.shape[-2] * recon.shape[-1]
        recon_norm = (recon ** 2).sum(dim=(-2, -1), keepdim=True).sqrt() + 1e-10
        target_norm = (target ** 2).sum(dim=(-2, -1), keepdim=True).sqrt() + 1e-10
        recon = recon / recon_norm * math.sqrt(sz)
        target = target / target_norm * math.sqrt(sz)
    elif normalize == 'sum':
        # Sum normalization
        sz = recon.shape[-2] * recon.shape[-1]
        recon = recon / (recon.sum(dim=(-2, -1), keepdim=True) + 1e-10) * sz
        target = target / (target.sum(dim=(-2, -1), keepdim=True) + 1e-10) * sz
    elif normalize == 'max':
        recon = recon / (recon.max() + 1e-10)
        target = target / (target.max() + 1e-10)
    # else 'none': no normalization

    if loss_type == 'L1':
        loss = torch.mean(torch.abs(recon - target))
    elif loss_type == 'L2':
        # Return RMSE (sqrt of MSE) to match original implementation
        # This is critical for gradient stability
        loss = torch.mean((recon - target) ** 2).sqrt()
    elif loss_type == 'Laplacian':
        # Laplacian edge-based loss
        kernel = torch.tensor(
            [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
            dtype=recon.dtype, device=recon.device
        ).view(1, 1, 3, 3)

        if recon.shape[1] > 1:
            kernel = kernel.repeat(recon.shape[1], 1, 1, 1)

        recon_lap = torch.nn.functional.conv2d(recon, kernel, padding=1, groups=recon.shape[1])
        target_lap = torch.nn.functional.conv2d(target, kernel, padding=1, groups=target.shape[1])
        loss = torch.mean((recon_lap - target_lap) ** 2)
    else:
        loss = torch.mean((recon - target) ** 2)

    return loss


def create_roi_mask(
    shape: tuple,
    roi_shape: tuple,
    device: torch.device = None
) -> torch.Tensor:
    """Create ROI (region of interest) mask.

    Args:
        shape: Full (H, W) shape
        roi_shape: ROI (H, W) shape (centered)
        device: Torch device

    Returns:
        Binary mask tensor [1, 1, H, W]
    """
    h, w = shape
    roi_h, roi_w = roi_shape

    mask = torch.zeros(1, 1, h, w, device=device)

    start_h = (h - roi_h) // 2
    start_w = (w - roi_w) // 2

    mask[:, :, start_h:start_h + roi_h, start_w:start_w + roi_w] = 1.0

    return mask


def cfftorder(n: int) -> np.ndarray:
    """Get centered FFT frequency order.

    Args:
        n: Number of samples

    Returns:
        Array of frequency indices centered at 0
    """
    if n % 2 == 0:
        return np.arange(-n//2, n//2)
    else:
        return np.arange(-(n-1)//2, (n+1)//2)
