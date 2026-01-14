"""
Evaluator class for DOE optimization results.

Provides unified evaluation interface for all DOE types.
"""

from typing import Optional, List, Tuple, Dict, Any, TYPE_CHECKING
from dataclasses import dataclass, field
import math
import numpy as np
import torch

from .metrics import EvaluationMetrics
from ..core.propagation import propagation_SFR

if TYPE_CHECKING:
    from ..params.base import StructuredParams
    from ..wizard.base import WizardOutput


class Evaluator:
    """Evaluator for DOE optimization results.

    Provides type-specific evaluation based on structured parameters.

    Example:
        evaluator = Evaluator(structured_params)
        metrics = evaluator.evaluate(
            target_intensity,
            simulated_intensity
        )
    """

    def __init__(
        self,
        structured_params: 'StructuredParams',
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Initialize evaluator.

        Args:
            structured_params: Structured parameters
            metadata: DOE-specific metadata (e.g., from wizard)
        """
        self.params = structured_params
        self.metadata = metadata or {}

    def evaluate(
        self,
        target: np.ndarray,
        simulated: np.ndarray,
        roi_mask: Optional[np.ndarray] = None
    ) -> EvaluationMetrics:
        """Evaluate optimization result.

        Args:
            target: Target intensity/amplitude pattern
            simulated: Simulated intensity/amplitude pattern
            roi_mask: Optional ROI mask

        Returns:
            EvaluationMetrics appropriate for the DOE type
        """
        doe_type = self._get_doe_type()

        if doe_type in ('splitter', 'spot_projector'):
            return self._evaluate_splitter(target, simulated)
        elif doe_type == 'diffuser':
            return self._evaluate_diffuser(target, simulated, roi_mask)
        elif doe_type in ('lens', 'lens_array'):
            return self._evaluate_lens(target, simulated)
        else:
            return self._evaluate_generic(target, simulated, roi_mask)

    def _get_doe_type(self) -> str:
        """Determine DOE type from params/metadata."""
        if 'grid_mode' in self.metadata:
            return 'splitter'
        if 'is_array' in self.metadata:
            return 'lens_array' if self.metadata['is_array'] else 'lens'
        if 'shape' in self.metadata:
            return 'diffuser'
        return 'custom'

    def _evaluate_splitter(
        self,
        target: np.ndarray,
        simulated: np.ndarray
    ) -> EvaluationMetrics:
        """Evaluate splitter results.

        Computes efficiency for each working order.
        """
        from ..params.fft_params import FFTParams
        from ..params.asm_params import ASMParams

        # Get working orders from metadata
        working_orders = self.metadata.get('working_orders', [])

        if not working_orders:
            # Fall back to peak detection
            return self._evaluate_peaks(simulated)

        # For FFT params, orders are at specific FFT positions
        if isinstance(self.params, FFTParams):
            period_h, period_w = self.params.period_pixels
            efficiencies = []
            labels = []

            # Total energy for normalization
            total_energy = (simulated ** 2).sum()

            for oy, ox in working_orders:
                # Order position in FFT output
                py = (period_h // 2 + oy) % period_h
                px = (period_w // 2 + ox) % period_w

                # Get efficiency (single pixel for delta target)
                if 0 <= py < period_h and 0 <= px < simulated.shape[-1]:
                    order_energy = simulated[py, px] ** 2
                    eff = float(order_energy / (total_energy + 1e-10))
                else:
                    eff = 0.0

                efficiencies.append(eff)
                labels.append(f"({oy},{ox})")

            return EvaluationMetrics.for_splitter(efficiencies, labels)

        # For ASM params, use spot detection
        return self._evaluate_peaks(simulated)

    def _evaluate_peaks(self, simulated: np.ndarray) -> EvaluationMetrics:
        """Evaluate by detecting intensity peaks."""
        from scipy import ndimage

        # Find local maxima
        data_max = ndimage.maximum_filter(simulated, size=5)
        peaks = (simulated == data_max) & (simulated > simulated.max() * 0.1)

        # Get peak positions and intensities
        peak_positions = np.argwhere(peaks)
        peak_intensities = simulated[peaks]

        total_energy = (simulated ** 2).sum()
        efficiencies = []
        labels = []

        for i, (y, x) in enumerate(peak_positions):
            eff = float(peak_intensities[i] ** 2 / (total_energy + 1e-10))
            efficiencies.append(eff)
            labels.append(f"peak_{i}")

        return EvaluationMetrics.for_splitter(efficiencies, labels)

    def _evaluate_diffuser(
        self,
        target: np.ndarray,
        simulated: np.ndarray,
        roi_mask: Optional[np.ndarray] = None
    ) -> EvaluationMetrics:
        """Evaluate diffuser results."""
        return EvaluationMetrics.for_diffuser(target, simulated, roi_mask)

    def _evaluate_lens(
        self,
        target: np.ndarray,
        simulated: np.ndarray
    ) -> EvaluationMetrics:
        """Evaluate lens results."""
        # Find focus center (maximum intensity)
        max_idx = np.unravel_index(np.argmax(simulated), simulated.shape)
        spot_center = (int(max_idx[0]), int(max_idx[1]))

        # Get Airy radius from metadata
        airy_radius = self.metadata.get('airy_radius_pixels', 3.0)

        return EvaluationMetrics.for_lens(simulated, spot_center, airy_radius)

    def _evaluate_generic(
        self,
        target: np.ndarray,
        simulated: np.ndarray,
        roi_mask: Optional[np.ndarray] = None
    ) -> EvaluationMetrics:
        """Generic evaluation for custom patterns."""
        return EvaluationMetrics.for_diffuser(target, simulated, roi_mask)


def evaluate_result(
    params: 'StructuredParams',
    target: np.ndarray,
    simulated: np.ndarray,
    metadata: Optional[Dict[str, Any]] = None,
    roi_mask: Optional[np.ndarray] = None
) -> EvaluationMetrics:
    """Convenience function for result evaluation.

    Args:
        params: Structured parameters
        target: Target pattern
        simulated: Simulated pattern
        metadata: DOE-specific metadata
        roi_mask: Optional ROI mask

    Returns:
        EvaluationMetrics
    """
    evaluator = Evaluator(params, metadata)
    return evaluator.evaluate(target, simulated, roi_mask)


@dataclass
class FiniteDistanceEvaluation:
    """Results from finite distance splitter evaluation with SFR propagation.

    This dataclass stores all data needed for visualization and analysis
    of finite distance splitter performance.

    Attributes:
        simulated_intensity: Intensity at target plane from SFR propagation
        target_intensity: Ideal target pattern (Gaussian spots)
        spot_positions_pixels: Spot positions in pixel coordinates [(py, px), ...]
        spot_positions_meters: Spot positions in physical coords [(y_m, x_m), ...]
        spot_efficiencies: Energy fraction within Airy disk for each spot
        airy_radius_pixels: Airy disk radius in pixels
        airy_radius_meters: Airy disk radius in meters
        output_size: Physical size of output (y, x) in meters
        output_pixel_size: Pixel size at output plane in meters
        output_resolution: Output resolution (H, W) in pixels
        working_orders: List of (oy, ox) diffraction order indices
        total_efficiency: Sum of all spot efficiencies
        mean_efficiency: Mean spot efficiency
        uniformity: 1 - (max - min) / (max + min)
    """
    simulated_intensity: np.ndarray
    target_intensity: np.ndarray
    spot_positions_pixels: List[Tuple[int, int]]
    spot_positions_meters: List[Tuple[float, float]]
    spot_efficiencies: List[float]
    airy_radius_pixels: float
    airy_radius_meters: float
    output_size: Tuple[float, float]
    output_pixel_size: float
    output_resolution: Tuple[int, int]
    working_orders: List[Tuple[int, int]]
    total_efficiency: float
    mean_efficiency: float
    uniformity: float


def evaluate_finite_distance_splitter(
    device_phase: np.ndarray,
    wavelength: float,
    pixel_size: float,
    working_distance: float,
    target_span: Tuple[float, float],
    working_orders: List[Tuple[int, int]],
    order_angles: List[Tuple[float, float]],
    fresnel_phase: Optional[np.ndarray] = None,
    output_resolution: Tuple[int, int] = (512, 512),
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
) -> FiniteDistanceEvaluation:
    """Evaluate finite distance splitter using SFR propagation with Airy disk integration.

    For Strategy 2 (Periodic + Fresnel) splitters, this function:
    1. Adds Fresnel lens phase to the device phase (if not provided)
    2. Uses SFR propagation to simulate at the target plane
    3. Calculates efficiency at each spot using Airy disk integration

    Args:
        device_phase: Full device phase array (2D, radians)
        wavelength: Wavelength in meters
        pixel_size: Pixel size in meters
        working_distance: Working distance in meters
        target_span: Target size (y, x) in meters
        working_orders: List of (oy, ox) order indices
        order_angles: List of (theta_y, theta_x) angles in radians
        fresnel_phase: Pre-computed Fresnel phase (optional)
        output_resolution: Output resolution for SFR (H, W)
        device: Torch device
        dtype: Torch dtype

    Returns:
        FiniteDistanceEvaluation with all results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    h, w = device_phase.shape
    device_size = (h * pixel_size, w * pixel_size)

    # Compute Fresnel lens phase if not provided
    if fresnel_phase is None:
        fresnel_phase = _compute_fresnel_phase(
            shape=(h, w),
            pixel_size=pixel_size,
            wavelength=wavelength,
            focal_length=working_distance
        )

    # Combine DOE phase with Fresnel lens
    combined_phase = device_phase + fresnel_phase

    # Create complex field
    field = np.exp(1j * combined_phase)
    field_tensor = torch.tensor(
        field.reshape(1, 1, h, w),
        dtype=torch.complex64,
        device=device
    )

    # SFR propagation to target plane
    output_field = propagation_SFR(
        u_in=field_tensor,
        feature_size=(pixel_size, pixel_size),
        wavelength=np.array([[[[wavelength]]]]),
        z=np.array([[[[working_distance]]]]),
        output_size=target_span,
        output_resolution=output_resolution,
        dtype=dtype
    )

    # Get intensity
    intensity = output_field.abs().squeeze().cpu().numpy() ** 2
    total_energy = intensity.sum()

    # Compute output pixel size
    out_h, out_w = output_resolution
    output_pixel_size = target_span[0] / out_h  # Assume square pixels

    # Compute Airy disk radius
    doe_diameter = min(h, w) * pixel_size
    airy_radius_m = 1.22 * wavelength * working_distance / doe_diameter
    airy_radius_px = airy_radius_m / output_pixel_size

    # Compute spot positions and efficiencies
    spot_positions_px = []
    spot_positions_m = []
    spot_efficiencies = []

    center_y = out_h // 2
    center_x = out_w // 2

    for theta_y, theta_x in order_angles:
        # Physical position
        pos_y_m = working_distance * math.tan(theta_y)
        pos_x_m = working_distance * math.tan(theta_x)

        # Pixel position
        py = int(center_y + pos_y_m / output_pixel_size)
        px = int(center_x + pos_x_m / output_pixel_size)

        spot_positions_px.append((py, px))
        spot_positions_m.append((pos_y_m, pos_x_m))

        # Compute efficiency within Airy disk
        eff = _compute_airy_disk_efficiency(
            intensity=intensity,
            center=(py, px),
            radius=airy_radius_px,
            total_energy=total_energy
        )
        spot_efficiencies.append(eff)

    # Compute aggregate metrics
    effs = np.array(spot_efficiencies)
    total_efficiency = float(effs.sum())
    mean_efficiency = float(effs.mean()) if len(effs) > 0 else 0.0

    if len(effs) > 0 and effs.max() + effs.min() > 1e-10:
        uniformity = 1.0 - (effs.max() - effs.min()) / (effs.max() + effs.min())
    else:
        uniformity = 0.0

    # Create target intensity (ideal Gaussian spots)
    target_intensity = _create_target_pattern(
        shape=output_resolution,
        spot_positions=spot_positions_px,
        spot_sigma=airy_radius_px / 2.355
    )

    return FiniteDistanceEvaluation(
        simulated_intensity=intensity,
        target_intensity=target_intensity,
        spot_positions_pixels=spot_positions_px,
        spot_positions_meters=spot_positions_m,
        spot_efficiencies=spot_efficiencies,
        airy_radius_pixels=airy_radius_px,
        airy_radius_meters=airy_radius_m,
        output_size=target_span,
        output_pixel_size=output_pixel_size,
        output_resolution=output_resolution,
        working_orders=working_orders,
        total_efficiency=total_efficiency,
        mean_efficiency=mean_efficiency,
        uniformity=float(uniformity)
    )


def _compute_fresnel_phase(
    shape: Tuple[int, int],
    pixel_size: float,
    wavelength: float,
    focal_length: float
) -> np.ndarray:
    """Compute Fresnel lens phase for focusing at given distance.

    Args:
        shape: (H, W) array shape
        pixel_size: Pixel size in meters
        wavelength: Wavelength in meters
        focal_length: Focal length (working distance) in meters

    Returns:
        Phase array in radians (wrapped to [0, 2*pi])
    """
    h, w = shape
    y = (np.arange(h) - h / 2) * pixel_size
    x = (np.arange(w) - w / 2) * pixel_size
    YY, XX = np.meshgrid(y, x, indexing='ij')

    # Fresnel lens phase (converging lens)
    k = 2 * np.pi / wavelength
    phase = -k / (2 * focal_length) * (XX ** 2 + YY ** 2)

    # Wrap to [0, 2*pi]
    phase = phase % (2 * np.pi)

    return phase


def _compute_airy_disk_efficiency(
    intensity: np.ndarray,
    center: Tuple[int, int],
    radius: float,
    total_energy: float
) -> float:
    """Compute energy fraction within Airy disk at given position.

    Args:
        intensity: 2D intensity array
        center: (py, px) center position in pixels
        radius: Airy disk radius in pixels
        total_energy: Total energy for normalization

    Returns:
        Efficiency (energy in disk / total energy)
    """
    h, w = intensity.shape
    cy, cx = center

    # Create circular mask
    y_coords = np.arange(h) - cy
    x_coords = np.arange(w) - cx
    YY, XX = np.meshgrid(y_coords, x_coords, indexing='ij')
    mask = (YY ** 2 + XX ** 2) <= radius ** 2

    # Sum energy in disk
    disk_energy = intensity[mask].sum()

    if total_energy > 1e-10:
        return float(disk_energy / total_energy)
    else:
        return 0.0


def _create_target_pattern(
    shape: Tuple[int, int],
    spot_positions: List[Tuple[int, int]],
    spot_sigma: float
) -> np.ndarray:
    """Create ideal target pattern with Gaussian spots.

    Args:
        shape: (H, W) output shape
        spot_positions: List of (py, px) spot centers
        spot_sigma: Gaussian sigma in pixels

    Returns:
        Target intensity pattern
    """
    h, w = shape
    target = np.zeros((h, w), dtype=np.float32)

    for py, px in spot_positions:
        if 0 <= py < h and 0 <= px < w:
            y_coords = np.arange(h) - py
            x_coords = np.arange(w) - px
            YY, XX = np.meshgrid(y_coords, x_coords, indexing='ij')
            gaussian = np.exp(-(YY ** 2 + XX ** 2) / (2 * spot_sigma ** 2))
            target += gaussian

    # Normalize
    if target.sum() > 0:
        target = target / target.sum()

    return target
