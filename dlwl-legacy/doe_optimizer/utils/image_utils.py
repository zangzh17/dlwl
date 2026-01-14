"""Image processing utility functions."""

import numpy as np
import torch
import torch.nn.functional as F


def pad_image(
    field: torch.Tensor,
    target_shape: tuple,
    padval: float = 0,
    mode: str = 'constant',
    pytorch: bool = True
) -> torch.Tensor:
    """Pad image to target shape.

    Args:
        field: Input tensor [B, C, H, W] or numpy array
        target_shape: Target (H, W) shape
        padval: Padding value for constant mode
        mode: Padding mode ('constant', 'wrap', 'reflect')
        pytorch: If True, input/output are torch tensors

    Returns:
        Padded tensor/array
    """
    if not pytorch:
        field = torch.from_numpy(field) if isinstance(field, np.ndarray) else field

    # Ensure 4D
    orig_ndim = field.ndim
    if field.ndim == 2:
        field = field.unsqueeze(0).unsqueeze(0)
    elif field.ndim == 3:
        field = field.unsqueeze(0)

    h, w = field.shape[-2:]
    target_h, target_w = target_shape[-2] if len(target_shape) > 1 else target_shape[0], \
                          target_shape[-1] if len(target_shape) > 1 else target_shape[0]

    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)

    # Symmetric padding
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if mode == 'constant':
        field = F.pad(field, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=padval)
    elif mode == 'wrap':
        # Circular padding
        field = F.pad(field, (pad_left, pad_right, pad_top, pad_bottom), mode='circular')
    else:
        field = F.pad(field, (pad_left, pad_right, pad_top, pad_bottom), mode=mode)

    # Restore original dimensions
    if orig_ndim == 2:
        field = field.squeeze(0).squeeze(0)
    elif orig_ndim == 3:
        field = field.squeeze(0)

    if not pytorch:
        return field.numpy()
    return field


def crop_image(
    field: torch.Tensor,
    target_shape: tuple,
    pytorch: bool = True
) -> torch.Tensor:
    """Crop image to target shape (center crop).

    Args:
        field: Input tensor [B, C, H, W] or numpy array
        target_shape: Target (H, W) shape
        pytorch: If True, input/output are torch tensors

    Returns:
        Cropped tensor/array
    """
    if not pytorch:
        field = torch.from_numpy(field) if isinstance(field, np.ndarray) else field

    # Ensure 4D
    orig_ndim = field.ndim
    if field.ndim == 2:
        field = field.unsqueeze(0).unsqueeze(0)
    elif field.ndim == 3:
        field = field.unsqueeze(0)

    h, w = field.shape[-2:]
    target_h = target_shape[-2] if len(target_shape) > 1 else target_shape[0]
    target_w = target_shape[-1] if len(target_shape) > 1 else target_shape[0]

    # Center crop
    start_h = (h - target_h) // 2
    start_w = (w - target_w) // 2

    field = field[..., start_h:start_h + target_h, start_w:start_w + target_w]

    # Restore original dimensions
    if orig_ndim == 2:
        field = field.squeeze(0).squeeze(0)
    elif orig_ndim == 3:
        field = field.squeeze(0)

    if not pytorch:
        return field.numpy()
    return field


def crop_pad_image(
    field: torch.Tensor,
    target_shape: tuple
) -> torch.Tensor:
    """Crop or pad image to target shape.

    Args:
        field: Input tensor [B, C, H, W]
        target_shape: Target (H, W) shape

    Returns:
        Resized tensor
    """
    h, w = field.shape[-2:]
    target_h, target_w = target_shape

    if h > target_h or w > target_w:
        field = crop_image(field, target_shape)
    if h < target_h or w < target_w:
        field = pad_image(field, target_shape)

    return field


def fft_interp(
    field: torch.Tensor,
    target_shape: tuple
) -> torch.Tensor:
    """Interpolate using FFT (zero-padding in frequency domain).

    Preserves frequency content while changing spatial resolution.

    Args:
        field: Input complex tensor [B, C, H, W]
        target_shape: Target (H, W) shape

    Returns:
        Interpolated tensor
    """
    h, w = field.shape[-2:]
    target_h, target_w = target_shape

    if h == target_h and w == target_w:
        return field

    # FFT
    F_field = torch.fft.fftshift(torch.fft.fft2(field), dim=(-2, -1))

    # Pad or crop in frequency domain
    if target_h > h or target_w > w:
        F_field = pad_image(F_field, target_shape, padval=0)
    else:
        F_field = crop_image(F_field, target_shape)

    # Inverse FFT with scaling
    scale = (target_h * target_w) / (h * w)
    field_out = torch.fft.ifft2(torch.fft.ifftshift(F_field, dim=(-2, -1))) * np.sqrt(scale)

    return field_out


def normalize(
    field: torch.Tensor,
    method: str = 'sqrt'
) -> torch.Tensor:
    """Normalize field amplitude.

    Args:
        field: Input tensor [B, C, H, W]
        method: Normalization method
            - 'sqrt': Normalize so sum of intensity = 1
            - 'max': Normalize to max = 1
            - 'energy': Normalize energy to 1

    Returns:
        Normalized tensor
    """
    if method == 'sqrt':
        # Normalize so that sum of |field|^2 = 1
        energy = torch.sum(torch.abs(field)**2, dim=(-2, -1), keepdim=True)
        return field / torch.sqrt(energy + 1e-10)
    elif method == 'max':
        max_val = torch.amax(torch.abs(field), dim=(-2, -1), keepdim=True)
        return field / (max_val + 1e-10)
    elif method == 'energy':
        energy = torch.sum(torch.abs(field)**2)
        return field / torch.sqrt(energy + 1e-10)
    else:
        return field


def upsample_nearest(
    field: torch.Tensor,
    scale_factor: int
) -> torch.Tensor:
    """Upsample using nearest neighbor interpolation.

    Args:
        field: Input tensor [B, C, H, W]
        scale_factor: Integer upsampling factor

    Returns:
        Upsampled tensor
    """
    if scale_factor == 1:
        return field
    return torch.kron(field, torch.ones(scale_factor, scale_factor, device=field.device, dtype=field.dtype))


def create_circular_mask(
    shape: tuple,
    device: torch.device = None
) -> torch.Tensor:
    """Create circular aperture mask.

    Args:
        shape: (H, W) mask shape
        device: Torch device

    Returns:
        Binary mask tensor [1, 1, H, W]
    """
    h, w = shape
    y = torch.linspace(-1, 1, h, device=device)
    x = torch.linspace(-1, 1, w, device=device)
    Y, X = torch.meshgrid(y, x, indexing='ij')
    mask = (X**2 + Y**2 <= 1).float()
    return mask.unsqueeze(0).unsqueeze(0)


def tile_to_size(
    pattern: np.ndarray,
    target_shape: tuple
) -> np.ndarray:
    """Tile a pattern to fill a larger target shape.

    Useful for tiling a DOE period to represent the full device.

    Args:
        pattern: Input array [H, W] (the period pattern)
        target_shape: Target (H, W) shape (full device size)

    Returns:
        Tiled array of shape target_shape
    """
    h, w = pattern.shape[:2]
    target_h, target_w = target_shape

    # Calculate number of tiles needed
    tiles_y = int(np.ceil(target_h / h))
    tiles_x = int(np.ceil(target_w / w))

    # Tile the pattern
    tiled = np.tile(pattern, (tiles_y, tiles_x))

    # Crop to exact target size
    return tiled[:target_h, :target_w]
