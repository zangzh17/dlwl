"""Two-step DOE optimization pipeline.

Step 1: Height/Phase optimization (traditional DOE design)
Step 2: Fabrication optimization (OPE correction) [optional]
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable
import numpy as np
import torch
import sys

from ..core.config import DOEConfig, DOEType, PropModel, FiniteDistanceStrategy
from ..core.propagation import get_propagator
from ..core.fabrication import FabricationModel, FitModel, create_fabrication_model
from ..core.optimizer import create_optimizer
from ..patterns.factory import generate_target_pattern
from ..utils.image_utils import upsample_nearest
from ..utils.math_utils import height2phase, spherical_phase
from .evaluation import evaluate_result, EvaluationMetrics


@dataclass
class DOEResult:
    """Result of DOE optimization.

    Attributes:
        height: Optimized height profile [H, W] in meters (may be period-sized for splitters)
        phase: Optimized phase profile [H, W] in radians (0 to 2pi)
        target_intensity: Target intensity pattern [H, W]
        simulated_intensity: Simulated intensity pattern [H, W]
        metrics: Evaluation metrics dictionary
        tolerance_limit: Computed tolerance limit (physical minimum)
        pixel_multiplier_options: Available pixel multiplier options

        # Full device representation
        device_height: Full device height profile (tiled from period if applicable) [H, W]
        device_phase: Full device phase profile [H, W]
        device_phase_with_fresnel: Full device phase with Fresnel overlay (for finite distance Strategy 2)
        period_pixels: Period size in pixels (for splitters) or None

        # Splitter-specific info (for visualization with order labels)
        splitter_params: Splitter parameters dict (mode, period, order_angles, etc.) or None

        # Fresnel phase overlay (for finite distance Strategy 2)
        fresnel_phase: Fresnel focusing phase [H, W] or None
        finite_distance_strategy: Strategy used for finite distance (ASM or PERIODIC_FRESNEL)

        # Step 2 results (if fab optimization enabled)
        fab_simulated_intensity: Simulated intensity after fab optimization [H, W]
        fab_metrics: Evaluation metrics after fab optimization
    """
    height: np.ndarray
    phase: np.ndarray
    target_intensity: np.ndarray
    simulated_intensity: np.ndarray
    metrics: EvaluationMetrics
    tolerance_limit: float
    pixel_multiplier_options: list

    # Full device representation
    device_height: Optional[np.ndarray] = None
    device_phase: Optional[np.ndarray] = None
    device_phase_with_fresnel: Optional[np.ndarray] = None
    period_pixels: Optional[int] = None

    # Splitter-specific info
    splitter_params: Optional[Dict[str, Any]] = None

    # Fresnel phase overlay (for finite distance Strategy 2)
    fresnel_phase: Optional[np.ndarray] = None
    finite_distance_strategy: Optional[FiniteDistanceStrategy] = None

    # Optional fabrication optimization results
    fab_simulated_intensity: Optional[np.ndarray] = None
    fab_metrics: Optional[EvaluationMetrics] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        result = {
            'height': self.height.tolist(),
            'phase': self.phase.tolist(),
            'target_intensity': self.target_intensity.tolist(),
            'simulated_intensity': self.simulated_intensity.tolist(),
            'metrics': self.metrics.to_dict(),
            'tolerance_limit': self.tolerance_limit,
            'pixel_multiplier_options': self.pixel_multiplier_options,
        }

        if self.device_height is not None:
            result['device_height'] = self.device_height.tolist()
        if self.device_phase is not None:
            result['device_phase'] = self.device_phase.tolist()
        if self.device_phase_with_fresnel is not None:
            result['device_phase_with_fresnel'] = self.device_phase_with_fresnel.tolist()
        if self.period_pixels is not None:
            result['period_pixels'] = self.period_pixels
        if self.splitter_params is not None:
            # Convert splitter_params, handling non-serializable types
            sp = self.splitter_params.copy()
            if 'mode' in sp:
                sp['mode'] = sp['mode'].value  # Convert enum to string
            if 'finite_distance_strategy' in sp and sp['finite_distance_strategy'] is not None:
                sp['finite_distance_strategy'] = sp['finite_distance_strategy'].value
            result['splitter_params'] = sp

        if self.fresnel_phase is not None:
            result['fresnel_phase'] = self.fresnel_phase.tolist()
        if self.finite_distance_strategy is not None:
            result['finite_distance_strategy'] = self.finite_distance_strategy.value

        if self.fab_simulated_intensity is not None:
            result['fab_simulated_intensity'] = self.fab_simulated_intensity.tolist()
        if self.fab_metrics is not None:
            result['fab_metrics'] = self.fab_metrics.to_dict()

        return result


def optimize_doe(
    config: DOEConfig,
    progress_callback: Optional[Callable[[str, int, float], None]] = None,
    device: torch.device = None,
    verbose: bool = True
) -> DOEResult:
    """Run DOE optimization pipeline.

    This is the main entry point for DOE optimization. It runs a two-step process:
    1. Height/Phase optimization to achieve target optical pattern
    2. (Optional) Fabrication optimization for OPE correction

    Args:
        config: DOEConfig instance with all parameters
        progress_callback: Optional callback(stage, iteration, loss) for progress updates
        device: Torch device (default: CUDA if available)
        verbose: If True, print progress messages

    Returns:
        DOEResult with optimized height, phase, simulated patterns, and metrics
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if verbose:
        print(f"Running DOE optimization on {device}")
        print(f"DOE type: {config.doe_type.value}")
        print(f"Propagation model: {config.prop_model.value}")
        print(f"Tolerance limit: {config.tolerance_limit:.4f} ({config.tolerance_limit * 100:.2f}%)")
        print(f"Max pixel multiplier: {config.max_pixel_multiplier}")

    # Generate target pattern
    if verbose:
        print("\n[Step 0] Generating target pattern...")
    target_amp = generate_target_pattern(config, device)
    target_intensity = (target_amp ** 2).squeeze().cpu().numpy()

    # Calculate height_2pi (height for 2pi phase shift)
    height_2pi = config.physical.height_2pi()
    if verbose:
        print(f"Height for 2pi phase: {height_2pi * 1e6:.3f} um")

    # Step 1: Height/Phase optimization
    if verbose:
        print("\n[Step 1] Running phase optimization...")
    height_optimized, recon_amp_phase, loss_phase = _run_phase_optimization(
        config, target_amp, height_2pi, device, progress_callback, verbose
    )

    # Convert height to phase
    wavelength = config.physical.wavelength_array
    refraction_index = config.physical.refraction_index_array
    phase_optimized = height2phase(
        height_optimized, wavelength, refraction_index
    ).squeeze(0).squeeze(0).cpu().numpy()

    # Ensure phase is in [0, 2pi] range
    # Note: Since height is clamped to [0, height_2pi], phase should already be in range
    # Using clip instead of mod to avoid wrapping values near 2pi back to ~0
    phase_optimized = np.clip(phase_optimized, 0, 2 * np.pi)

    # Get height in numpy
    # Use squeeze(0).squeeze(0) to only remove batch and channel dims, preserving 2D spatial dims
    height_np = height_optimized.squeeze(0).squeeze(0).cpu().numpy()

    # Simulated intensity from step 1
    sim_intensity_phase = (recon_amp_phase ** 2).squeeze(0).squeeze(0).cpu().numpy()

    # Crop recon_amp to match target size for evaluation
    from ..utils.image_utils import crop_image
    # For splitters, use splitter resolution; otherwise use phase_resolution
    if config.is_splitter():
        roi_res = config.get_splitter_resolution()
    else:
        roi_res = config.target.roi_resolution or config.phase_resolution
    recon_amp_roi = crop_image(recon_amp_phase, roi_res)
    target_amp_roi = crop_image(target_amp, roi_res)

    # Evaluate step 1 results
    if verbose:
        print("\n[Step 1] Evaluating results...")
    metrics_phase = evaluate_result(
        config, target_amp_roi.squeeze().cpu().numpy(),
        recon_amp_roi.squeeze().cpu().numpy()
    )
    if verbose:
        print(f"Step 1 Loss: {loss_phase:.6f}")
        print(f"Step 1 Metrics: {metrics_phase}")

    # Step 2: Fabrication optimization (optional)
    fab_sim_intensity = None
    fab_metrics = None

    if config.enable_fab_optimization and config.fabrication is not None:
        if verbose:
            print("\n[Step 2] Running fabrication optimization...")

        # Upsample height if pixel multiplier > 1
        if config.optimization.phase_pixel_multiplier > 1:
            height_upsampled = upsample_nearest(
                height_optimized,
                config.optimization.phase_pixel_multiplier
            )
        else:
            height_upsampled = height_optimized

        # Add height offset for fabrication
        fab_model = create_fabrication_model(
            config.fabrication,
            config.get_feature_size(for_phase=False),
            device
        )
        depth_max = float(fab_model.depth_max)
        height_min = (depth_max - height_2pi) / 2
        height_for_fab = height_upsampled + height_min

        # Run fabrication optimization
        dose_optimized, height_fab, recon_amp_fab, loss_fab = _run_fab_optimization(
            config, height_for_fab, target_amp, fab_model, device, progress_callback, verbose
        )

        fab_sim_intensity = (recon_amp_fab ** 2).squeeze().cpu().numpy()

        # Evaluate step 2 results
        if verbose:
            print("\n[Step 2] Evaluating results...")
        fab_metrics = evaluate_result(
            config, target_amp.squeeze().cpu().numpy(),
            recon_amp_fab.squeeze().cpu().numpy()
        )
        if verbose:
            print(f"Step 2 Loss: {loss_fab:.6f}")
            print(f"Step 2 Metrics: {fab_metrics}")

    # Compute full device representation
    device_height = None
    device_phase = None
    device_phase_with_fresnel = None
    period_pixels = None
    splitter_params = None
    fresnel_phase = None
    finite_strategy = None

    if config.is_splitter():
        # For splitters, tile the period to full device size
        from ..utils.image_utils import tile_to_size
        device_res = config.slm_resolution

        # Get splitter-specific parameters
        splitter_params = config.get_splitter_params()
        finite_strategy = config.get_finite_distance_strategy()

        # The optimized pattern is now at period_pixels resolution (e.g., 27×27)
        # which matches the physical period exactly. No resampling needed!
        period_meters = config.get_splitter_period()
        period_pixels = int(round(period_meters / config.device.pixel_size))

        if finite_strategy == FiniteDistanceStrategy.ASM:
            # Strategy 1: Non-periodic, phase IS the full device
            device_height = height_np
            device_phase = phase_optimized
        else:
            # Infinite distance or Strategy 2: tile the period
            # Optimization resolution now equals period_pixels, no resampling needed
            height_period = height_np
            phase_period = phase_optimized

            # Tile the period to device resolution
            device_height = tile_to_size(height_period, device_res)
            device_phase = tile_to_size(phase_period, device_res)

            # For Strategy 2 (PERIODIC_FRESNEL), add Fresnel focusing phase
            if finite_strategy == FiniteDistanceStrategy.PERIODIC_FRESNEL:
                z = splitter_params.get('working_distance')
                if z is not None:
                    # Generate Fresnel phase for focusing at working distance
                    fresnel_phase_tensor = spherical_phase(
                        shape=device_res,
                        feature_size=(config.device.pixel_size, config.device.pixel_size),
                        wavelength=config.physical.wavelength_array,
                        focal_length=z,
                        dtype=torch.float32
                    )
                    fresnel_phase = fresnel_phase_tensor.squeeze().numpy()

                    # Combine grating phase with Fresnel phase (mod 2pi)
                    device_phase_with_fresnel = np.mod(
                        device_phase + fresnel_phase,
                        2 * np.pi
                    )
    else:
        # For non-splitters, the optimized height/phase IS the device
        device_height = height_np
        device_phase = phase_optimized

    # Build result
    result = DOEResult(
        height=height_np,
        phase=phase_optimized,
        target_intensity=target_intensity,
        simulated_intensity=sim_intensity_phase,
        metrics=metrics_phase,
        tolerance_limit=config.tolerance_limit,
        pixel_multiplier_options=config.get_pixel_multiplier_options(),
        device_height=device_height,
        device_phase=device_phase,
        device_phase_with_fresnel=device_phase_with_fresnel,
        period_pixels=period_pixels,
        splitter_params=splitter_params,
        fresnel_phase=fresnel_phase,
        finite_distance_strategy=finite_strategy,
        fab_simulated_intensity=fab_sim_intensity,
        fab_metrics=fab_metrics,
    )

    if verbose:
        print("\nOptimization complete!")
    return result


def _run_phase_optimization(
    config: DOEConfig,
    target_amp: torch.Tensor,
    height_2pi: float,
    device: torch.device,
    progress_callback: Optional[Callable] = None,
    verbose: bool = True
) -> tuple:
    """Run Step 1: Phase/Height optimization.

    Args:
        config: DOE configuration
        target_amp: Target amplitude tensor
        height_2pi: Height for 2pi phase shift
        device: Torch device
        progress_callback: Progress callback
        verbose: If True, print progress

    Returns:
        Tuple of (height_optimized, recon_amp, final_loss)
    """
    method = config.optimization.phase_method
    params = config.optimization

    # Create optimizer for phase optimization
    optimizer = create_optimizer(
        method=method,
        config=config,
        fab_model=None,  # No fab model for phase optimization
        for_phase=True,
        device=device
    )

    # Initialize random height
    # For splitters, use splitter-specific resolution (small, matching diffraction orders)
    # Exception: Strategy 1 (ASM) uses full device resolution for non-periodic optimization
    if config.is_splitter():
        from ..core.config import FiniteDistanceStrategy
        strategy = config.get_finite_distance_strategy()
        if strategy == FiniteDistanceStrategy.ASM:
            # Strategy 1: Use full device resolution for non-periodic ASM optimization
            resolution = config.phase_resolution
        else:
            # Infinite distance or Strategy 2: Use small splitter resolution (periodic)
            resolution = config.get_splitter_resolution()
    else:
        resolution = config.phase_resolution
    init_height = height_2pi * torch.rand(1, 1, *resolution, device=device)

    # Callback wrapper with progress reporting
    def callback(iter_num, loss):
        if progress_callback:
            progress_callback("phase", iter_num, loss)
        if verbose:
            # Print progress every 100 iterations
            print(f"\r  Iter {iter_num:5d}/{params.phase_iters}, Loss: {loss:.6f}", end="")
            sys.stdout.flush()

    # Run optimization
    height_opt, recon_amp, loss = optimizer.optimize(
        target=target_amp,
        init_value=init_height,
        num_iters=params.phase_iters,
        lr=params.phase_lr,
        min_value=0.0,
        max_value=height_2pi,
        loss_type=params.loss_type,
        optimizer_type=params.optimizer_type,
        upsample_factor=params.simulation_upsample,
        progress_callback=callback
    )

    if verbose:
        print()  # Newline after progress

    # Clamp to valid range
    height_opt = torch.clamp(height_opt, 0.0, height_2pi)

    return height_opt, recon_amp, loss


def _run_fab_optimization(
    config: DOEConfig,
    target_height: torch.Tensor,
    target_amp: torch.Tensor,
    fab_model: FabricationModel,
    device: torch.device,
    progress_callback: Optional[Callable] = None,
    verbose: bool = True
) -> tuple:
    """Run Step 2: Fabrication optimization.

    Args:
        config: DOE configuration
        target_height: Target height from step 1
        target_amp: Original target amplitude
        fab_model: Fabrication model
        device: Torch device
        progress_callback: Progress callback
        verbose: If True, print progress

    Returns:
        Tuple of (dose_optimized, height_out, recon_amp, final_loss)
    """
    method = config.optimization.fab_method
    params = config.optimization

    # Create optimizer for fab optimization
    optimizer = create_optimizer(
        method=method,
        config=config,
        fab_model=fab_model,
        for_phase=False,
        device=device
    )

    # Initialize dose using inverse GT
    init_dose = fab_model.gt_inv(target_height)
    init_dose = torch.clamp(init_dose, 0.0, 255.0)

    # Callback wrapper with progress reporting
    def callback(iter_num, loss):
        if progress_callback:
            progress_callback("fab", iter_num, loss)
        if verbose:
            print(f"\r  Iter {iter_num:5d}/{params.fab_iters}, Loss: {loss:.6f}", end="")
            sys.stdout.flush()

    # Run optimization (optimize dose to match target height)
    dose_opt, recon_amp, loss = optimizer.optimize(
        target=target_amp,  # Use original target for optical loss
        init_value=init_dose,
        num_iters=params.fab_iters,
        lr=params.fab_lr,
        min_value=0.0,
        max_value=255.0,
        loss_type=params.loss_type,
        optimizer_type=params.optimizer_type,
        progress_callback=callback
    )

    if verbose:
        print()  # Newline after progress

    # Get final height from dose
    height_out = fab_model(dose_opt, backward=False)

    return dose_opt, height_out, recon_amp, loss
