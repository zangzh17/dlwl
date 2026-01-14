"""
Unified re-evaluation module for DOE optimization results.

Provides consistent upsampling/re-propagation interface for all propagation types.

Key concepts:
- Upsample factor k means k^2 more samples covering the SAME physical/angular range
- All propagators: nearest-neighbor upsample phase (each pixel → k×k block, same value)
- This is equivalent to kron operation for integer upsampling
- Output always covers the same range, just with more samples
"""

from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class ReevaluationResult:
    """Result of re-evaluation at different resolution.

    Attributes:
        simulated_intensity: Re-propagated intensity [H_new, W_new]
        target_intensity: Target pattern at new resolution [H_new, W_new]
        phase: Phase used for simulation [H_new, W_new] (nearest-neighbor upsampled)
        metrics: Computed efficiency metrics
        upsample_factor: Applied upsample factor
        effective_pixel_size: Simulation pixel size (original / upsample_factor)
    """
    simulated_intensity: np.ndarray
    target_intensity: np.ndarray
    phase: np.ndarray
    metrics: Dict[str, Any]
    upsample_factor: int
    effective_pixel_size: float


def reevaluate_at_resolution(
    phase: np.ndarray,
    target: np.ndarray,
    upsample_factor: int,
    propagation_type: str,
    pixel_size: float,
    wavelength: float = 532e-9,
    working_distance: Optional[float] = None,
    target_size: Optional[Tuple[float, float]] = None,
    target_indices: Optional[List[Tuple[int, int]]] = None,
    device: torch.device = None
) -> ReevaluationResult:
    """Re-evaluate optimization result at different resolution.

    Unified interface for all propagation types. The output always covers
    the same physical/angular range, just with more samples.

    Args:
        phase: Original optimized phase [H, W] (single period for FFT)
        target: Original target pattern [H, W]
        upsample_factor: Resolution multiplier (1 = original, 2 = 2x, etc.)
        propagation_type: 'fft', 'asm', 'sfr', or 'periodic_fresnel'
        pixel_size: Original physical pixel size in meters
        wavelength: Wavelength in meters
        working_distance: Propagation distance for ASM/SFR (meters)
        target_size: Physical target size for SFR (meters)
        target_indices: Target positions in original coordinates [(y, x), ...]
        device: Torch device (default: CPU for stability)

    Returns:
        ReevaluationResult with re-propagated data and metrics
    """
    if device is None:
        device = torch.device('cpu')

    if upsample_factor <= 1:
        # No upsampling - return original with metrics
        return ReevaluationResult(
            simulated_intensity=np.abs(np.fft.fftshift(np.fft.fft2(np.exp(1j * phase)))) ** 2
                if propagation_type == 'fft' else phase,  # Placeholder
            target_intensity=target,
            phase=phase,
            metrics=_compute_metrics(target, target_indices),
            upsample_factor=1,
            effective_pixel_size=pixel_size
        )

    # Effective pixel size decreases with upsampling
    effective_pixel_size = pixel_size / upsample_factor

    if propagation_type in ('fft', 'periodic_fresnel'):
        return _reevaluate_fft(
            phase, target, upsample_factor, target_indices, device, effective_pixel_size
        )
    elif propagation_type == 'asm':
        return _reevaluate_asm(
            phase, target, upsample_factor, pixel_size, wavelength,
            working_distance, target_size, target_indices, device, effective_pixel_size
        )
    elif propagation_type == 'sfr':
        return _reevaluate_sfr(
            phase, target, upsample_factor, pixel_size, wavelength,
            working_distance, target_size, target_indices, device, effective_pixel_size
        )
    else:
        raise ValueError(f"Unknown propagation type: {propagation_type}")


def _reevaluate_fft(
    phase: np.ndarray,
    target: np.ndarray,
    upsample_factor: int,
    target_indices: Optional[List[Tuple[int, int]]],
    device: torch.device,
    effective_pixel_size: float
) -> ReevaluationResult:
    """Re-evaluate FFT result with kron-upsampled phase.

    Kron upsampling simulates a DOE with finer fabrication pixels:
    - Same period, more pixels per period
    - k-space range expands k times, but we CROP back to original range
    - Output size stays the same as original (independent of upsample)
    - Values change due to sinc envelope from smaller pixel apertures
    """
    from scipy import ndimage

    original_shape = phase.shape
    is_1d = phase.ndim == 1 or (phase.ndim == 2 and min(phase.shape) == 1)
    h_orig, w_orig = original_shape if len(original_shape) == 2 else (1, original_shape[0])

    # Kron upsample phase (each pixel → k×k block, same value)
    if is_1d:
        phase_upsampled = ndimage.zoom(phase.reshape(-1), upsample_factor, order=0)
    else:
        phase_upsampled = ndimage.zoom(phase, upsample_factor, order=0)

    # FFT propagation using KRON-UPSAMPLED phase
    phase_tensor = torch.tensor(phase_upsampled, dtype=torch.float32, device=device)
    if phase_tensor.ndim == 1:
        phase_tensor = phase_tensor.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    elif phase_tensor.ndim == 2:
        phase_tensor = phase_tensor.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        field = torch.exp(1j * phase_tensor.to(torch.complex64))
        output = torch.fft.fftshift(
            torch.fft.fftn(
                torch.fft.fftshift(field, dim=(-2, -1)),
                dim=(-2, -1),
                norm="ortho"
            ),
            dim=(-2, -1)
        )
        simulated_full = output.abs() ** 2

    simulated_full_np = simulated_full.squeeze().cpu().numpy()

    # CROP to original size (same k-space range as original)
    # Key: DC component must stay at the same index after cropping
    # For fftshifted output: DC is at N//2 for both odd and even N
    # Correct alignment: start = upsampled_center - original_center
    if is_1d:
        w_full = len(simulated_full_np)
        cx_full = w_full // 2  # DC index in full output
        cx_orig = w_orig // 2  # DC index in original
        start = cx_full - cx_orig
        simulated_np = simulated_full_np[start:start + w_orig]
    else:
        h_full, w_full = simulated_full_np.shape
        cy_full, cx_full = h_full // 2, w_full // 2  # DC indices in full output
        cy_orig, cx_orig = h_orig // 2, w_orig // 2  # DC indices in original
        start_h = cy_full - cy_orig
        start_w = cx_full - cx_orig
        simulated_np = simulated_full_np[start_h:start_h + h_orig, start_w:start_w + w_orig]

    simulated_np = simulated_np / (simulated_np.sum() + 1e-10)

    # Target stays at ORIGINAL size (indices unchanged)
    if is_1d:
        target_new = np.zeros(w_orig, dtype=np.float32)
        for py, px in (target_indices or []):
            if 0 <= px < len(target_new):
                target_new[px] = 1.0
    else:
        target_new = np.zeros((h_orig, w_orig), dtype=np.float32)
        for py, px in (target_indices or []):
            if 0 <= py < target_new.shape[0] and 0 <= px < target_new.shape[1]:
                target_new[py, px] = 1.0

    target_new = target_new / (target_new.sum() + 1e-10)

    # Compute metrics using ORIGINAL indices (output is same size as original)
    metrics = _compute_fft_metrics(simulated_np, target_indices or [], is_1d)

    return ReevaluationResult(
        simulated_intensity=simulated_np,
        target_intensity=target_new,
        phase=phase_upsampled,
        metrics=metrics,
        upsample_factor=upsample_factor,
        effective_pixel_size=effective_pixel_size
    )


def _reevaluate_asm(
    phase: np.ndarray,
    target: np.ndarray,
    upsample_factor: int,
    pixel_size: float,
    wavelength: float,
    working_distance: float,
    target_size: Optional[Tuple[float, float]],
    target_indices: Optional[List[Tuple[int, int]]],
    device: torch.device,
    effective_pixel_size: float
) -> ReevaluationResult:
    """Re-evaluate ASM result with upsampled phase (finer DOE pixels).

    Analysis Upsample for ASM (same as SFR):
    - Upsample the INPUT PHASE (each pixel -> k×k block with same value)
    - Use smaller effective_pixel_size = pixel_size / upsample_factor
    - OUTPUT resolution and physical range stay UNCHANGED
    - This simulates a DOE with finer fabrication pixels covering same area
    """
    from scipy import ndimage
    from ..core.propagation import propagation_ASM

    original_shape = target.shape  # Use target shape as output resolution
    h_orig, w_orig = original_shape

    # Upsample phase: each pixel -> k×k block with same value (nearest neighbor)
    # This simulates finer DOE pixels covering the same physical area
    phase_upsampled = ndimage.zoom(phase, upsample_factor, order=0)

    # Effective pixel size decreases (finer pixels)
    # DOE physical size stays the same
    eff_pixel_size = pixel_size / upsample_factor

    # Prepare upsampled phase tensor
    phase_tensor = torch.tensor(phase_upsampled, dtype=torch.float32, device=device)
    phase_tensor = phase_tensor.unsqueeze(0).unsqueeze(0)

    # ASM propagation with:
    # - Upsampled phase (more pixels)
    # - Smaller feature_size (effective_pixel_size)
    # - Same target_size (same physical output area)
    # - ORIGINAL output_resolution (same number of output samples)
    with torch.no_grad():
        field = torch.exp(1j * phase_tensor.to(torch.complex64))
        output = propagation_ASM(
            u_in=field,
            feature_size=(eff_pixel_size, eff_pixel_size),  # Smaller pixel size
            wavelength=np.array([[[[wavelength]]]]),
            z=np.array([[[[working_distance]]]]),
            output_size=target_size,  # Same physical area
            output_resolution=(h_orig, w_orig),  # ORIGINAL output resolution
            linear_conv=True
        )
        simulated = output.abs() ** 2

    simulated_np = simulated.squeeze().cpu().numpy()
    simulated_np = simulated_np / (simulated_np.sum() + 1e-10)

    # Target stays at ORIGINAL size (output resolution unchanged)
    target_normalized = target / (target.sum() + 1e-10)

    # Target indices stay the same (output grid unchanged)
    # Use upsample_factor=1 for metrics since output resolution is original
    metrics = _compute_asm_metrics(simulated_np, target_normalized, target_indices or [], upsample_factor=1)

    return ReevaluationResult(
        simulated_intensity=simulated_np,
        target_intensity=target_normalized,
        phase=phase_upsampled,  # Return upsampled phase
        metrics=metrics,
        upsample_factor=upsample_factor,
        effective_pixel_size=eff_pixel_size
    )


def _reevaluate_sfr(
    phase: np.ndarray,
    target: np.ndarray,
    upsample_factor: int,
    pixel_size: float,
    wavelength: float,
    working_distance: float,
    target_size: Tuple[float, float],
    target_indices: Optional[List[Tuple[int, int]]],
    device: torch.device,
    effective_pixel_size: float
) -> ReevaluationResult:
    """Re-evaluate SFR result with upsampled phase (finer DOE pixels).

    Analysis Upsample for SFR:
    - Upsample the INPUT PHASE (each pixel -> k×k block with same value)
    - Use smaller effective_pixel_size = pixel_size / upsample_factor
    - OUTPUT resolution and physical range stay UNCHANGED
    - This simulates a DOE with finer fabrication pixels covering same area

    The purpose is to verify the effect of input sampling resolution
    on the diffraction pattern.
    """
    from scipy import ndimage
    from ..core.propagation import propagation_SFR

    original_shape = target.shape
    h_orig, w_orig = original_shape

    # Upsample phase: each pixel -> k×k block with same value (nearest neighbor)
    # This simulates finer DOE pixels covering the same physical area
    phase_upsampled = ndimage.zoom(phase, upsample_factor, order=0)

    # Effective pixel size decreases (finer pixels)
    # DOE physical size stays the same: (h_orig * pixel_size) = (h_up * effective_pixel_size)
    eff_pixel_size = pixel_size / upsample_factor

    # Prepare upsampled phase tensor
    phase_tensor = torch.tensor(phase_upsampled, dtype=torch.float32, device=device)
    phase_tensor = phase_tensor.unsqueeze(0).unsqueeze(0)

    # SFR propagation with:
    # - Upsampled phase (more pixels)
    # - Smaller feature_size (effective_pixel_size)
    # - Same target_size (same physical output area)
    # - ORIGINAL output_resolution (same number of output samples)
    with torch.no_grad():
        field = torch.exp(1j * phase_tensor.to(torch.complex64))
        output = propagation_SFR(
            u_in=field,
            feature_size=(eff_pixel_size, eff_pixel_size),  # Smaller pixel size
            wavelength=np.array([[[[wavelength]]]]),
            z=np.array([[[[working_distance]]]]),
            output_size=target_size,  # Same physical area
            output_resolution=(h_orig, w_orig)  # ORIGINAL output resolution
        )
        simulated = output.abs() ** 2

    simulated_np = simulated.squeeze().cpu().numpy()
    simulated_np = simulated_np / (simulated_np.sum() + 1e-10)

    # Target stays at ORIGINAL size (output resolution unchanged)
    target_normalized = target / (target.sum() + 1e-10)

    # Target indices stay the same (output grid unchanged)
    # Use upsample_factor=1 for metrics since output resolution is original
    metrics = _compute_asm_metrics(simulated_np, target_normalized, target_indices or [], upsample_factor=1)

    return ReevaluationResult(
        simulated_intensity=simulated_np,
        target_intensity=target_normalized,
        phase=phase_upsampled,  # Return upsampled phase
        metrics=metrics,
        upsample_factor=upsample_factor,
        effective_pixel_size=eff_pixel_size
    )


def _scale_indices_for_tiling(
    indices: Optional[List[Tuple[int, int]]],
    original_shape: Tuple[int, ...],
    new_shape: Tuple[int, ...],
    num_tiles: int,
    is_1d: bool
) -> List[Tuple[int, int]]:
    """Scale target indices for tiled FFT output.

    In tiled FFT, diffraction orders stay at same relative positions
    but scaled from center.
    """
    if not indices:
        return []

    scaled = []

    if is_1d:
        w_orig = original_shape[0] if len(original_shape) == 1 else original_shape[-1]
        w_new = new_shape[0] if len(new_shape) == 1 else new_shape[-1]
        cx_orig = w_orig // 2
        cx_new = w_new // 2

        for py, px in indices:
            dx = px - cx_orig
            new_px = cx_new + dx * num_tiles
            scaled.append((0, int(new_px)))
    else:
        h_orig, w_orig = original_shape[:2]
        h_new, w_new = new_shape[:2]
        cy_orig, cx_orig = h_orig // 2, w_orig // 2
        cy_new, cx_new = h_new // 2, w_new // 2

        for py, px in indices:
            dy = py - cy_orig
            dx = px - cx_orig
            new_py = cy_new + dy * num_tiles
            new_px = cx_new + dx * num_tiles
            scaled.append((int(new_py), int(new_px)))

    return scaled


def _compute_metrics(
    target: np.ndarray,
    target_indices: Optional[List[Tuple[int, int]]]
) -> Dict[str, Any]:
    """Compute basic metrics."""
    return {
        'num_targets': len(target_indices) if target_indices else 0
    }


def _compute_fft_metrics(
    simulated: np.ndarray,
    target_indices: List[Tuple[int, int]],
    is_1d: bool
) -> Dict[str, Any]:
    """Compute efficiency metrics for FFT output."""
    if not target_indices:
        return {'total_efficiency': 0.0, 'uniformity': 0.0, 'order_efficiencies': []}

    total_energy = simulated.sum()
    efficiencies = []

    for py, px in target_indices:
        if is_1d:
            if 0 <= px < len(simulated):
                eff = float(simulated[px] / (total_energy + 1e-10))
            else:
                eff = 0.0
        else:
            if 0 <= py < simulated.shape[0] and 0 <= px < simulated.shape[1]:
                eff = float(simulated[py, px] / (total_energy + 1e-10))
            else:
                eff = 0.0
        efficiencies.append(eff)

    effs = np.array(efficiencies)
    total_eff = float(effs.sum())
    mean_eff = float(effs.mean()) if len(effs) > 0 else 0.0

    if len(effs) > 0 and effs.max() + effs.min() > 1e-10:
        uniformity = 1.0 - (effs.max() - effs.min()) / (effs.max() + effs.min())
    else:
        uniformity = 0.0

    return {
        'total_efficiency': total_eff,
        'uniformity': float(uniformity),
        'mean_efficiency': mean_eff,
        'std_efficiency': float(effs.std()) if len(effs) > 0 else 0.0,
        'order_efficiencies': efficiencies
    }


def _compute_asm_metrics(
    simulated: np.ndarray,
    target: np.ndarray,
    target_indices: List[Tuple[int, int]],
    upsample_factor: int = 1
) -> Dict[str, Any]:
    """Compute metrics for ASM/SFR output using Airy disk integration.

    Args:
        simulated: Simulated intensity [H, W]
        target: Target intensity [H, W]
        target_indices: Target positions [(y, x), ...] in upsampled coordinates
        upsample_factor: Resolution multiplier (used to scale integration radius)

    Returns:
        Dictionary with efficiency metrics
    """
    # Simple MSE-based metric for now
    mse = float(np.mean((simulated - target) ** 2))

    # Efficiency at target positions (with small radius integration)
    # The integration radius should scale with upsample_factor to cover
    # the same physical area regardless of resolution
    total_energy = simulated.sum()
    efficiencies = []
    base_integration_radius = 3  # pixels at 1x
    integration_radius = base_integration_radius * upsample_factor

    for py, px in target_indices:
        h, w = simulated.shape
        y_coords = np.arange(h) - py
        x_coords = np.arange(w) - px
        YY, XX = np.meshgrid(y_coords, x_coords, indexing='ij')
        mask = (YY ** 2 + XX ** 2) <= integration_radius ** 2

        if mask.any():
            eff = float(simulated[mask].sum() / (total_energy + 1e-10))
        else:
            eff = 0.0
        efficiencies.append(eff)

    effs = np.array(efficiencies) if efficiencies else np.array([0.0])

    return {
        'total_efficiency': float(effs.sum()),
        'uniformity': 1.0 - (effs.max() - effs.min()) / (effs.max() + effs.min() + 1e-10),
        'mean_efficiency': float(effs.mean()),
        'std_efficiency': float(effs.std()),
        'order_efficiencies': efficiencies,
        'mse': mse
    }


def extract_target_indices(target: np.ndarray, threshold: float = 1e-6) -> List[Tuple[int, int]]:
    """Extract target point indices from target pattern.

    Args:
        target: Target intensity pattern
        threshold: Minimum value to consider as target

    Returns:
        List of (y, x) positions
    """
    if target is None:
        return []

    # Handle 1D case
    if target.ndim == 1 or (target.ndim == 2 and min(target.shape) == 1):
        target_1d = target.reshape(-1)
        indices = np.where(target_1d > threshold)[0]
        return [(0, int(i)) for i in indices]

    # 2D case
    positions = np.where(target > threshold)
    return [(int(py), int(px)) for py, px in zip(positions[0], positions[1])]
