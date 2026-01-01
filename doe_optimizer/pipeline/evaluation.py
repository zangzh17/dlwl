"""Evaluation metrics for DOE optimization results."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple, List
import numpy as np

from ..core.config import DOEConfig, DOEType


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for DOE optimization.

    Different metrics are computed based on DOE type:
    - Splitter/Deflector/Spot Projector: order efficiency and uniformity
    - Diffuser/Custom: PSNR and SSIM
    - Lens/Lens Array: encircled energy ratio
    """
    # Common metrics
    total_efficiency: float = 0.0  # Total energy in target region

    # Order-based metrics (splitter, deflector, spot projector)
    order_efficiencies: Optional[List[float]] = None
    order_efficiency_mean: Optional[float] = None
    order_efficiency_std: Optional[float] = None
    order_uniformity: Optional[float] = None  # 1 - std/mean

    # Pattern metrics (diffuser, custom)
    psnr: Optional[float] = None
    ssim: Optional[float] = None

    # Lens metrics
    encircled_energy: Optional[float] = None  # Energy within Airy disk
    airy_radius_pixels: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {'total_efficiency': self.total_efficiency}

        if self.order_efficiencies is not None:
            result['order_efficiencies'] = self.order_efficiencies
        if self.order_efficiency_mean is not None:
            result['order_efficiency_mean'] = self.order_efficiency_mean
        if self.order_efficiency_std is not None:
            result['order_efficiency_std'] = self.order_efficiency_std
        if self.order_uniformity is not None:
            result['order_uniformity'] = self.order_uniformity
        if self.psnr is not None:
            result['psnr'] = self.psnr
        if self.ssim is not None:
            result['ssim'] = self.ssim
        if self.encircled_energy is not None:
            result['encircled_energy'] = self.encircled_energy
        if self.airy_radius_pixels is not None:
            result['airy_radius_pixels'] = self.airy_radius_pixels

        return result

    def __str__(self) -> str:
        """Human-readable string representation."""
        parts = [f"Efficiency: {self.total_efficiency:.4f}"]

        if self.order_efficiency_mean is not None:
            parts.append(f"Order Mean: {self.order_efficiency_mean:.4f}")
            parts.append(f"Order Std: {self.order_efficiency_std:.4f}")
            parts.append(f"Uniformity: {self.order_uniformity:.4f}")

        if self.psnr is not None:
            parts.append(f"PSNR: {self.psnr:.2f} dB")
            parts.append(f"SSIM: {self.ssim:.4f}")

        if self.encircled_energy is not None:
            parts.append(f"Encircled Energy: {self.encircled_energy:.4f}")

        return ", ".join(parts)


def evaluate_result(
    config: DOEConfig,
    target_amp: np.ndarray,
    recon_amp: np.ndarray
) -> EvaluationMetrics:
    """Evaluate optimization result based on DOE type.

    Args:
        config: DOE configuration
        target_amp: Target amplitude [H, W]
        recon_amp: Reconstructed amplitude [H, W]

    Returns:
        EvaluationMetrics with computed metrics
    """
    doe_type = config.doe_type

    # Compute total efficiency
    target_intensity = target_amp ** 2
    recon_intensity = recon_amp ** 2

    # Normalize reconstructed intensity
    recon_intensity = recon_intensity / (recon_intensity.sum() + 1e-10)
    target_mask = target_intensity > 0.01 * target_intensity.max()
    total_efficiency = float(recon_intensity[target_mask].sum())

    metrics = EvaluationMetrics(total_efficiency=total_efficiency)

    # Type-specific metrics
    if doe_type in [DOEType.SPLITTER_1D, DOEType.SPLITTER_2D,
                    DOEType.DEFLECTOR, DOEType.SPOT_PROJECTOR]:
        _compute_order_metrics(config, target_intensity, recon_intensity, metrics)

    elif doe_type in [DOEType.DIFFUSER, DOEType.CUSTOM]:
        _compute_pattern_metrics(target_amp, recon_amp, metrics)

    elif doe_type in [DOEType.LENS, DOEType.LENS_ARRAY]:
        _compute_lens_metrics(config, target_intensity, recon_intensity, metrics)

    return metrics


def _compute_order_metrics(
    config: DOEConfig,
    target_intensity: np.ndarray,
    recon_intensity: np.ndarray,
    metrics: EvaluationMetrics
) -> None:
    """Compute order-based metrics for splitter/deflector/spot projector.

    Args:
        config: DOE configuration
        target_intensity: Target intensity pattern
        recon_intensity: Reconstructed intensity pattern
        metrics: EvaluationMetrics to update
    """
    # Find target order positions (peaks in target)
    threshold = 0.1 * target_intensity.max()
    order_mask = target_intensity > threshold

    # For splitters, each pixel is a separate order (don't use connected regions)
    # This handles the case where target pixels are adjacent
    if config.is_splitter():
        # Handle 1D splitters where squeeze() may have removed the second dimension
        if target_intensity.ndim == 1:
            # 1D case: only y indices
            order_positions_y = np.where(order_mask)[0]
            num_orders = len(order_positions_y)

            if num_orders == 0:
                return

            efficiencies = []
            recon_norm = recon_intensity / (recon_intensity.sum() + 1e-10)

            for y in order_positions_y:
                order_efficiency = float(recon_norm[y])
                efficiencies.append(order_efficiency)
        else:
            # 2D case
            order_positions = np.where(order_mask)
            num_orders = len(order_positions[0])

            if num_orders == 0:
                return

            efficiencies = []
            recon_norm = recon_intensity / (recon_intensity.sum() + 1e-10)

            for i in range(num_orders):
                y, x = order_positions[0][i], order_positions[1][i]
                order_efficiency = float(recon_norm[y, x])
                efficiencies.append(order_efficiency)
    else:
        # For non-splitters, use connected region labeling
        from scipy import ndimage
        labeled, num_orders = ndimage.label(order_mask)

        if num_orders == 0:
            return

        efficiencies = []
        for i in range(1, num_orders + 1):
            order_region = labeled == i
            # Sum reconstructed intensity in this order region
            # Expand region slightly for integration
            order_region_dilated = ndimage.binary_dilation(order_region, iterations=2)
            order_efficiency = float(recon_intensity[order_region_dilated].sum())
            efficiencies.append(order_efficiency)

    efficiencies = np.array(efficiencies)

    metrics.order_efficiencies = efficiencies.tolist()
    metrics.order_efficiency_mean = float(efficiencies.mean())
    metrics.order_efficiency_std = float(efficiencies.std())

    if metrics.order_efficiency_mean > 0:
        metrics.order_uniformity = float(
            1.0 - metrics.order_efficiency_std / metrics.order_efficiency_mean
        )
    else:
        metrics.order_uniformity = 0.0


def _compute_pattern_metrics(
    target_amp: np.ndarray,
    recon_amp: np.ndarray,
    metrics: EvaluationMetrics
) -> None:
    """Compute pattern metrics (PSNR, SSIM) for diffuser/custom.

    Args:
        target_amp: Target amplitude
        recon_amp: Reconstructed amplitude
        metrics: EvaluationMetrics to update
    """
    # Normalize for comparison
    target_norm = target_amp / (target_amp.max() + 1e-10)
    recon_norm = recon_amp / (recon_amp.max() + 1e-10)

    # PSNR
    mse = np.mean((target_norm - recon_norm) ** 2)
    if mse > 0:
        metrics.psnr = float(10 * np.log10(1.0 / mse))
    else:
        metrics.psnr = float('inf')

    # SSIM
    metrics.ssim = float(_compute_ssim(target_norm, recon_norm))


def _compute_ssim(
    img1: np.ndarray,
    img2: np.ndarray,
    k1: float = 0.01,
    k2: float = 0.03,
    win_size: int = 7
) -> float:
    """Compute SSIM between two images.

    Args:
        img1: First image
        img2: Second image
        k1, k2: SSIM constants
        win_size: Window size for local statistics

    Returns:
        SSIM value
    """
    from scipy.ndimage import uniform_filter

    c1 = k1 ** 2
    c2 = k2 ** 2

    # Local means
    mu1 = uniform_filter(img1, size=win_size, mode='reflect')
    mu2 = uniform_filter(img2, size=win_size, mode='reflect')

    # Local variances and covariance
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = uniform_filter(img1 ** 2, size=win_size, mode='reflect') - mu1_sq
    sigma2_sq = uniform_filter(img2 ** 2, size=win_size, mode='reflect') - mu2_sq
    sigma12 = uniform_filter(img1 * img2, size=win_size, mode='reflect') - mu1_mu2

    # SSIM formula
    numerator = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    denominator = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)

    ssim_map = numerator / (denominator + 1e-10)
    return float(ssim_map.mean())


def _compute_lens_metrics(
    config: DOEConfig,
    target_intensity: np.ndarray,
    recon_intensity: np.ndarray,
    metrics: EvaluationMetrics
) -> None:
    """Compute lens metrics (encircled energy) for lens/lens array.

    Args:
        config: DOE configuration
        target_intensity: Target intensity pattern
        recon_intensity: Reconstructed intensity pattern
        metrics: EvaluationMetrics to update
    """
    h, w = recon_intensity.shape

    # Find focus position (peak of target)
    peak_idx = np.argmax(target_intensity)
    peak_y, peak_x = np.unravel_index(peak_idx, target_intensity.shape)

    # Calculate Airy disk radius in pixels
    # Airy radius = 1.22 * lambda * f / D
    # In output plane: r_airy = 1.22 * lambda * z / D
    wavelength = config.physical._wavelength_arr.mean()
    pixel_size = config.device.pixel_size
    z = config.physical.working_distance

    # Approximate aperture as DOE size
    D = min(h, w) * pixel_size

    # Handle infinite distance case
    if z is None or z == float('inf'):
        # For far-field, use a fixed reasonable radius
        airy_radius_m = wavelength / D  # Approximate
    else:
        # Airy radius in meters
        airy_radius_m = 1.22 * wavelength * z / D

    # Convert to pixels (in output plane)
    # For SFR, output pixel size might differ
    output_pixel_size = pixel_size  # Approximate
    airy_radius_px = airy_radius_m / output_pixel_size

    # Ensure minimum radius
    airy_radius_px = max(airy_radius_px, 2.0)
    metrics.airy_radius_pixels = float(airy_radius_px)

    # Create circular mask around peak
    y_coords, x_coords = np.ogrid[:h, :w]
    dist_sq = (y_coords - peak_y) ** 2 + (x_coords - peak_x) ** 2
    airy_mask = dist_sq <= airy_radius_px ** 2

    # Compute encircled energy
    total_energy = recon_intensity.sum()
    encircled = recon_intensity[airy_mask].sum()

    if total_energy > 0:
        metrics.encircled_energy = float(encircled / total_energy)
    else:
        metrics.encircled_energy = 0.0


def compute_diffraction_efficiency(
    recon_intensity: np.ndarray,
    target_positions: List[Tuple[int, int]],
    integration_radius: int = 3
) -> Tuple[float, List[float]]:
    """Compute diffraction efficiency at specific positions.

    Args:
        recon_intensity: Reconstructed intensity pattern
        target_positions: List of (y, x) positions for target orders
        integration_radius: Radius for integration around each position

    Returns:
        Tuple of (total_efficiency, list of individual efficiencies)
    """
    h, w = recon_intensity.shape
    total_energy = recon_intensity.sum()

    if total_energy == 0:
        return 0.0, [0.0] * len(target_positions)

    y_coords, x_coords = np.ogrid[:h, :w]
    efficiencies = []

    for py, px in target_positions:
        dist_sq = (y_coords - py) ** 2 + (x_coords - px) ** 2
        mask = dist_sq <= integration_radius ** 2
        eff = float(recon_intensity[mask].sum() / total_energy)
        efficiencies.append(eff)

    total_eff = sum(efficiencies)
    return total_eff, efficiencies


@dataclass
class FiniteDistanceEvaluation:
    """Evaluation results for finite distance splitter using propagation simulation.

    Contains both the simulated intensity at target plane and per-spot efficiencies.
    """
    simulated_intensity: np.ndarray  # Intensity at target plane [H, W]
    target_intensity: np.ndarray     # Target pattern (ideal spots) [H, W]
    spot_efficiencies: List[float]   # Efficiency for each spot (energy in Airy disk)
    spot_positions_pixels: List[Tuple[int, int]]  # Spot positions in pixels
    spot_positions_meters: List[Tuple[float, float]]  # Spot positions in meters
    airy_radius_pixels: float        # Airy disk radius in pixels
    airy_radius_meters: float        # Airy disk radius in meters
    total_efficiency: float          # Sum of all spot efficiencies
    mean_efficiency: float           # Mean spot efficiency
    uniformity: float                # 1 - std/mean
    output_pixel_size: Tuple[float, float]  # Pixel size at target plane (dy, dx)
    output_size: Tuple[float, float]        # Physical size of target plane
    working_orders: Optional[List[Tuple[int, int]]] = None  # Order indices for labeling


def evaluate_finite_distance_splitter(
    device_phase: np.ndarray,
    config: DOEConfig,
    splitter_params: Dict[str, Any],
    use_fresnel_phase: bool = True,
    fresnel_phase: Optional[np.ndarray] = None,
    output_resolution: Optional[Tuple[int, int]] = None,
    output_scale: float = 1.5,  # Scale factor for output area (1.5 = 50% margin)
    upsample_factor: int = 2,  # Upsampling factor for more accurate propagation
    optimized_phase: Optional[np.ndarray] = None,  # Original optimized phase (for k-space evaluation)
) -> FiniteDistanceEvaluation:
    """Evaluate splitter at finite working distance using propagation simulation.

    For Strategy 1 (ASM): Direct propagation, output size ≈ input size
    For Strategy 2 (Periodic + Fresnel): Use SFR with target output size

    Args:
        device_phase: Device phase profile [H, W] in radians
        config: DOE configuration
        splitter_params: Splitter parameters from config.get_splitter_params()
        use_fresnel_phase: If True and Strategy 2, add Fresnel phase overlay
        fresnel_phase: Pre-computed Fresnel phase (optional)
        output_resolution: Override output resolution for simulation
        output_scale: Scale factor for output area (1.5 = 50% margin)
        upsample_factor: Upsampling factor for propagation (default 2)

    Returns:
        FiniteDistanceEvaluation with simulated intensity and spot efficiencies
    """
    import torch
    from ..core.propagation import propagation_ASM, propagation_SFR
    from ..core.config import FiniteDistanceStrategy
    from ..utils.image_utils import fft_interp

    strategy = splitter_params.get('finite_distance_strategy')
    z = splitter_params.get('working_distance')

    if z is None:
        raise ValueError("Cannot evaluate finite distance for infinite working distance")

    wavelength = config.physical.wavelength_array
    pixel_size = config.device.pixel_size
    D = config.device.diameter  # DOE aperture

    # Compute Airy disk radius: r_airy = 1.22 * lambda * z / D
    wavelength_scalar = config.physical._wavelength_arr.mean()
    airy_radius_m = 1.22 * wavelength_scalar * z / D

    # Prepare input field (uniform amplitude with device phase)
    h, w = device_phase.shape
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Apply Fresnel phase for Strategy 2 if requested
    if strategy == FiniteDistanceStrategy.PERIODIC_FRESNEL and use_fresnel_phase:
        if fresnel_phase is not None:
            total_phase = device_phase + fresnel_phase
        else:
            # Generate Fresnel phase
            from ..utils.math_utils import spherical_phase
            fresnel_tensor = spherical_phase(
                shape=(h, w),
                feature_size=(pixel_size, pixel_size),
                wavelength=wavelength,
                focal_length=z,
            )
            total_phase = device_phase + fresnel_tensor.squeeze().numpy()
        total_phase = np.mod(total_phase, 2 * np.pi)
    else:
        total_phase = device_phase

    # Create input complex field
    u_in = torch.exp(1j * torch.tensor(total_phase, dtype=torch.float32)).unsqueeze(0).unsqueeze(0)
    u_in = u_in.to(device)

    # Apply upsampling for more accurate propagation simulation
    if upsample_factor > 1:
        upsampled_res = (h * upsample_factor, w * upsample_factor)
        u_in = fft_interp(u_in, upsampled_res)
        # Update effective pixel size after upsampling
        effective_pixel_size = pixel_size / upsample_factor
        h_up, w_up = upsampled_res
    else:
        effective_pixel_size = pixel_size
        h_up, w_up = h, w

    # Get target positions (physical, in meters)
    target_positions_m = splitter_params.get('target_positions', [])
    if not target_positions_m:
        raise ValueError("No target positions available for finite distance evaluation")
    target_span = config.target.target_span  # (x, y) physical size

    # Determine output size and resolution
    if strategy == FiniteDistanceStrategy.ASM:
        # Strategy 1: ASM - output size same as input (physical size unchanged)
        output_size_m = (h * pixel_size, w * pixel_size)
        out_res = (h, w) if output_resolution is None else output_resolution
        output_pixel_size = (output_size_m[0] / out_res[0], output_size_m[1] / out_res[1])

        # Propagate using ASM with upsampled input
        z_arr = np.array([[[[z]]]])
        u_out = propagation_ASM(
            u_in,
            feature_size=(effective_pixel_size, effective_pixel_size),
            wavelength=wavelength,
            z=z_arr,
            output_resolution=out_res,  # Downsample back to original resolution
        )
    else:
        # Strategy 2: Periodic + Fresnel
        # For a periodic DOE with Fresnel lens overlay, the physical interpretation is:
        # - The periodic DOE creates diffraction orders at angles θ_m = arcsin(m*λ/Λ)
        # - The Fresnel lens focuses each order to position x_m = z * tan(θ_m) ≈ z * m*λ/Λ
        #
        # The FFT output IS the correct k-space pattern - each pixel corresponds to one
        # diffraction order, NOT a physical position.
        #
        # IMPORTANT: For k-space FFT evaluation, we use ONLY the device phase (no Fresnel).
        # The Fresnel phase is a physical overlay for focusing but doesn't change k-space pattern.
        from ..core.propagation import propagation_FFT

        # For k-space evaluation, use the original optimized phase if provided
        # This gives the exact same pattern as during optimization
        if optimized_phase is not None:
            phase_for_eval = optimized_phase
        else:
            # Fallback: extract one period from device_phase
            period = splitter_params.get('period', h * pixel_size)
            period_pixels = int(round(period / pixel_size))

            # Extract one period from center of device_phase
            center_y, center_x = h // 2, w // 2
            start_y = center_y - period_pixels // 2
            start_x = center_x - period_pixels // 2
            end_y = start_y + period_pixels
            end_x = start_x + period_pixels

            # Ensure we stay in bounds
            start_y = max(0, start_y)
            start_x = max(0, start_x)
            end_y = min(h, end_y)
            end_x = min(w, end_x)

            # Extract one period
            if end_y - start_y == period_pixels and end_x - start_x == period_pixels:
                phase_for_eval = device_phase[start_y:end_y, start_x:end_x]
            else:
                # Fallback: use first period_pixels x period_pixels
                phase_for_eval = device_phase[:period_pixels, :period_pixels]

        # Create input (no Fresnel phase for k-space evaluation)
        u_in_kspace = torch.exp(1j * torch.tensor(phase_for_eval, dtype=torch.float32)).unsqueeze(0).unsqueeze(0)
        u_in_kspace = u_in_kspace.to(device)

        # Use FFT to compute the far-field pattern (same as optimization)
        u_out_kspace = propagation_FFT(u_in_kspace)
        intensity_kspace = (u_out_kspace.abs() ** 2).squeeze().cpu().numpy()

        # Resolution for order mapping matches the phase used
        h_up_kspace, w_up_kspace = phase_for_eval.shape

        # Normalize
        total_energy_kspace = intensity_kspace.sum()
        if total_energy_kspace > 0:
            intensity_kspace = intensity_kspace / total_energy_kspace

        # Get working orders directly from splitter params (more reliable than reverse-engineering)
        working_orders = splitter_params.get('working_orders', None)
        period = splitter_params.get('period', h * pixel_size)

        # Calculate spot efficiencies directly from k-space pattern
        # Since we're using original optimization resolution, order mapping is 1:1
        center_y_kspace, center_x_kspace = h_up_kspace // 2, w_up_kspace // 2

        # Period for position computation
        period_y = period
        period_x = period

        # Compute efficiency for each working order directly
        spot_efficiencies = []
        spot_positions_pixels_final = []
        target_positions_m_final = []

        if working_orders is not None:
            # Use working_orders directly (most reliable)
            for my, mx in working_orders:
                # Map order to pixel position in k-space (1:1 mapping since using one period)
                iy = center_y_kspace + my
                ix = center_x_kspace + mx

                if 0 <= iy < h_up_kspace and 0 <= ix < w_up_kspace:
                    eff = float(intensity_kspace[iy, ix])
                else:
                    eff = 0.0

                # Compute physical position for visualization
                pos_y = z * my * wavelength_scalar / period_y
                pos_x = z * mx * wavelength_scalar / period_x

                spot_efficiencies.append(eff)
                target_positions_m_final.append((pos_y, pos_x))
        else:
            # Fallback: reverse-engineer from target_positions_m
            for pos_y, pos_x in target_positions_m:
                my = round(pos_y * period_y / (z * wavelength_scalar))
                mx = round(pos_x * period_x / (z * wavelength_scalar))

                # Map order to pixel position (1:1 mapping)
                iy = center_y_kspace + my
                ix = center_x_kspace + mx

                if 0 <= iy < h_up_kspace and 0 <= ix < w_up_kspace:
                    eff = float(intensity_kspace[iy, ix])
                else:
                    eff = 0.0

                spot_efficiencies.append(eff)
                target_positions_m_final.append((pos_y, pos_x))

        # For visualization, create an output image showing spot positions
        # Use a grid where each spot is visualized at its physical position
        output_size_m = (target_span[0] * output_scale, target_span[1] * output_scale)
        out_res = output_resolution if output_resolution else (512, 512)
        output_pixel_size = (output_size_m[0] / out_res[0], output_size_m[1] / out_res[1])

        # Create visualization image
        intensity_vis = np.zeros((out_res[0], out_res[1]), dtype=np.float32)
        out_h, out_w = out_res

        for i, (pos_y, pos_x) in enumerate(target_positions_m_final):
            py = int(out_h / 2 + pos_y / output_pixel_size[0])
            px = int(out_w / 2 + pos_x / output_pixel_size[1])
            if 0 <= py < out_h and 0 <= px < out_w:
                # Draw a spot with intensity proportional to efficiency
                spot_positions_pixels_final.append((py, px))
                # Create Gaussian spot for visualization
                y_grid, x_grid = np.ogrid[:out_h, :out_w]
                sigma = airy_radius_m / output_pixel_size[0] / 2
                sigma = max(sigma, 2.0)
                dist_sq = (y_grid - py) ** 2 + (x_grid - px) ** 2
                intensity_vis += spot_efficiencies[i] * np.exp(-dist_sq / (2 * sigma ** 2))

        # Normalize visualization
        if intensity_vis.max() > 0:
            intensity_vis = intensity_vis / intensity_vis.max()

        # Use the k-space intensity for actual evaluation, visualization for display
        intensity = intensity_vis
        target_positions_m = target_positions_m_final
        spot_positions_pixels = spot_positions_pixels_final

        # Compute statistics from k-space efficiencies
        eff_array = np.array(spot_efficiencies)
        total_eff = float(eff_array.sum())
        mean_eff = float(eff_array.mean()) if len(eff_array) > 0 else 0.0
        std_eff = float(eff_array.std()) if len(eff_array) > 0 else 0.0
        uniformity = 1.0 - std_eff / mean_eff if mean_eff > 0 else 0.0

        # Return early for Strategy 2 since we computed everything already
        return FiniteDistanceEvaluation(
            simulated_intensity=intensity,
            target_intensity=np.zeros_like(intensity),  # Not used for k-space eval
            spot_efficiencies=spot_efficiencies,
            spot_positions_pixels=spot_positions_pixels,
            spot_positions_meters=target_positions_m,
            airy_radius_pixels=float(airy_radius_m / output_pixel_size[0]),
            airy_radius_meters=float(airy_radius_m),
            total_efficiency=total_eff,
            mean_efficiency=mean_eff,
            uniformity=uniformity,
            output_pixel_size=output_pixel_size,
            output_size=output_size_m,
            working_orders=working_orders,
        )

    # Get intensity
    intensity = (u_out.abs() ** 2).squeeze().cpu().numpy()
    total_energy = intensity.sum()

    # Normalize intensity for numerical stability
    if total_energy > 0:
        intensity = intensity / total_energy
        total_energy = 1.0

    # Find intensity peaks for debugging
    from scipy.ndimage import maximum_filter
    local_max = (intensity == maximum_filter(intensity, size=20))
    peak_mask = local_max & (intensity > 0.001)  # Threshold at 0.1% of normalized
    peaks_idx = np.where(peak_mask)
    n_peaks = len(peaks_idx[0])

    # Convert target positions to pixel coordinates
    out_h, out_w = intensity.shape
    spot_positions_pixels = []

    # The Fresnel/SFR propagation maps diffraction angle θ to output position z*sin(θ)
    # This is NOT inverted - spots at positive angles appear at positive positions.
    # No sign flip is needed.
    corrected_positions_m = list(target_positions_m)

    for pos_y, pos_x in corrected_positions_m:
        # Convert from physical position to pixel (centered coordinate system)
        py = int(out_h / 2 + pos_y / output_pixel_size[0])
        px = int(out_w / 2 + pos_x / output_pixel_size[1])
        # Clamp to valid range
        py = max(0, min(out_h - 1, py))
        px = max(0, min(out_w - 1, px))
        spot_positions_pixels.append((py, px))

    # Update target_positions_m to corrected version for return value
    target_positions_m = corrected_positions_m

    # Compute Airy radius in pixels
    airy_radius_px = airy_radius_m / output_pixel_size[0]
    airy_radius_px = max(airy_radius_px, 2.0)  # Minimum 2 pixels

    # Generate target pattern (Gaussian spots at target positions)
    target_intensity = np.zeros((out_h, out_w), dtype=np.float32)
    y_grid, x_grid = np.ogrid[:out_h, :out_w]
    spot_sigma = airy_radius_px / 2.0  # Gaussian sigma (half of Airy radius)

    for py, px in spot_positions_pixels:
        # Create Gaussian spot
        dist_sq = (y_grid - py) ** 2 + (x_grid - px) ** 2
        target_intensity += np.exp(-dist_sq / (2 * spot_sigma ** 2))

    # Normalize target pattern
    if target_intensity.max() > 0:
        target_intensity = target_intensity / target_intensity.max()

    # Compute spot efficiencies (energy within Airy disk)
    y_coords, x_coords = np.ogrid[:out_h, :out_w]
    spot_efficiencies = []

    for py, px in spot_positions_pixels:
        dist_sq = (y_coords - py) ** 2 + (x_coords - px) ** 2
        airy_mask = dist_sq <= airy_radius_px ** 2
        if total_energy > 0:
            eff = float(intensity[airy_mask].sum() / total_energy)
        else:
            eff = 0.0
        spot_efficiencies.append(eff)

    # Compute statistics
    eff_array = np.array(spot_efficiencies)
    total_eff = float(eff_array.sum())
    mean_eff = float(eff_array.mean()) if len(eff_array) > 0 else 0.0
    std_eff = float(eff_array.std()) if len(eff_array) > 0 else 0.0
    uniformity = 1.0 - std_eff / mean_eff if mean_eff > 0 else 0.0

    # Get working orders for labeling
    working_orders = splitter_params.get('working_orders', None)

    return FiniteDistanceEvaluation(
        simulated_intensity=intensity,
        target_intensity=target_intensity,
        spot_efficiencies=spot_efficiencies,
        spot_positions_pixels=spot_positions_pixels,
        spot_positions_meters=target_positions_m,
        airy_radius_pixels=float(airy_radius_px),
        airy_radius_meters=float(airy_radius_m),
        total_efficiency=total_eff,
        mean_efficiency=mean_eff,
        uniformity=uniformity,
        output_pixel_size=output_pixel_size,
        output_size=output_size_m,
        working_orders=working_orders,
    )
