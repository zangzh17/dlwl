"""
OptimizationRunner - New unified entry point for DOE optimization.

This module provides the main orchestrator that uses the layered architecture:
1. Wizard layer for parameter generation
2. Validation layer for constraint checking
3. Propagator and Loss function creation
4. Optimization execution
5. Evaluation and visualization
"""

from typing import Optional, Callable, Dict, Any, List, Tuple
from dataclasses import dataclass
import torch
import numpy as np

from ..api.request import OptimizationRequest
from ..api.response import OptimizationResponse, DOEResultData, MetricsData
from ..api.errors import ValidationError, ErrorCode
from ..wizard.factory import create_wizard, generate_params
from ..wizard.base import WizardOutput
from ..validation.validator import StructuredParamsValidator
from ..validation.messages import ValidationResult
from ..params.base import PropagationType, StructuredParams
from ..params.fft_params import FFTParams
from ..params.asm_params import ASMParams
from ..params.sfr_params import SFRParams
from ..core.propagator_factory import PropagatorBuilder
from ..core.loss import create_loss, BaseLoss, L2Loss
from ..visualization.data import (
    VisualizationData,
    create_phase_heatmap,
    create_intensity_heatmap,
    create_efficiency_bar_chart
)
from .progress import ProgressReporter, CancellationToken, ProgressInfo


@dataclass
class OptimizationResult:
    """Internal optimization result before conversion to API format."""
    height: np.ndarray
    phase: np.ndarray
    target_intensity: np.ndarray
    simulated_intensity: np.ndarray
    final_loss: float
    wizard_output: WizardOutput
    # For metrics calculation with pixel_multiplier
    simulated_for_metrics: np.ndarray = None
    target_for_metrics: np.ndarray = None
    reduced_resolution: tuple = None
    original_resolution: tuple = None
    pixel_multiplier: int = 1


class OptimizationRunner:
    """Main orchestrator for DOE optimization using layered architecture.

    This class coordinates:
    1. Parameter generation (via Wizard)
    2. Parameter validation
    3. Propagator and loss function creation
    4. Optimizer creation and execution
    5. Result evaluation
    6. Visualization data generation

    Example:
        runner = OptimizationRunner()

        request = OptimizationRequest.from_json({
            'doe_type': 'splitter_2d',
            'wavelength': 532e-9,
            'device_diameter': 1e-3,
            'pixel_size': 0.5e-6,
            'target_spec': {
                'num_spots': [5, 5],
                'target_type': 'angle',
                'target_span': [0.1, 0.1],
                'grid_mode': 'natural'
            }
        })

        response = runner.run(request, progress_callback=print_progress)
    """

    def __init__(
        self,
        max_resolution: int = 2000,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32
    ):
        """Initialize optimization runner.

        Args:
            max_resolution: Maximum simulation resolution
            device: Torch device
            dtype: Torch dtype
        """
        self.max_resolution = max_resolution
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        self.validator = StructuredParamsValidator(max_resolution=max_resolution)

    def run(
        self,
        request: OptimizationRequest,
        progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
        cancellation_token: Optional[CancellationToken] = None
    ) -> OptimizationResponse:
        """Run complete optimization pipeline.

        Args:
            request: Optimization request
            progress_callback: Optional progress callback
            cancellation_token: Optional cancellation token

        Returns:
            OptimizationResponse with results or errors
        """
        try:
            # 1. Generate parameters via Wizard (or use provided structured params/target_pattern)
            if request.has_structured_params:
                # Advanced user: use provided structured params directly
                wizard_output = self._create_wizard_output_from_params(
                    request.structured_params
                )
            elif request.user_input.get('target_pattern') is not None:
                # Target pattern provided directly (stored from previous wizard call)
                # Build minimal wizard output without calling wizard
                wizard_output = self._create_wizard_output_from_target_pattern(
                    request.user_input
                )
            else:
                # Normal flow: use wizard to generate params
                wizard_output = generate_params(
                    request.user_input,
                    max_resolution=self.max_resolution
                )

                # Apply DOE Settings overrides from frontend (if present)
                # This allows user-edited DOE Settings to override wizard defaults
                wizard_output = self._apply_doe_settings_overrides(
                    wizard_output,
                    request.user_input
                )

            # 2. Validate parameters
            validation = self.validator.validate(wizard_output.structured_params)
            if not validation.is_valid:
                return OptimizationResponse.error_response(
                    session_id=request.session_id,
                    errors=[ValidationError(
                        code=ErrorCode(msg.code),
                        message=msg.message,
                        field=msg.field,
                        suggestion=msg.suggestion
                    ) for msg in validation.errors],
                    warnings=[w for w in wizard_output.warnings]
                )

            # 3. Run optimization
            result = self._run_optimization(
                wizard_output=wizard_output,
                progress_callback=progress_callback,
                cancellation_token=cancellation_token
            )

            if result is None:
                # Cancelled
                return OptimizationResponse(
                    success=False,
                    session_id=request.session_id,
                    errors=[ValidationError(
                        code=ErrorCode.COMPUTATION_LIMIT,
                        message="Optimization cancelled by user"
                    )]
                )

            # 4. Create result data
            result_data = self._create_result_data(result, wizard_output)

            # 5. Create visualization data
            visualization = self._create_visualization_data(result, wizard_output)

            return OptimizationResponse.success_response(
                session_id=request.session_id,
                result=result_data,
                visualization=visualization,
                warnings=wizard_output.warnings
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return OptimizationResponse.error_response(
                session_id=request.session_id,
                errors=[ValidationError(
                    code=ErrorCode.COMPUTATION_LIMIT,
                    message=str(e)
                )]
            )

    def _create_wizard_output_from_params(
        self,
        params: StructuredParams
    ) -> WizardOutput:
        """Create WizardOutput from pre-computed structured params."""
        from ..params.optimization import OptimizationConfig

        propagator_config = params.to_propagator_config()

        return WizardOutput(
            structured_params=params,
            propagator_config=propagator_config,
            optimization_config=OptimizationConfig(),
            target_pattern=params.target_pattern,
            computed_values={},
            warnings=[],
            metadata={}
        )

    def _create_wizard_output_from_target_pattern(
        self,
        user_input: Dict[str, Any]
    ) -> WizardOutput:
        """Create WizardOutput from stored target_pattern and DOE Settings.

        This bypasses the wizard completely, using DOE Settings from frontend.
        The target_pattern was already generated by wizard and stored in frontend.

        Args:
            user_input: Dict containing target_pattern and DOE Settings

        Returns:
            WizardOutput ready for optimization
        """
        from ..params.optimization import OptimizationConfig, OptMethod, LossType, LossConfig
        from ..params.base import PropagationType, PropagatorConfig, PhysicalConstants

        # Extract target pattern from stored data
        target_pattern_list = user_input['target_pattern']
        target_np = np.array(target_pattern_list, dtype=np.float32)
        target_tensor = torch.from_numpy(target_np).unsqueeze(0).unsqueeze(0)

        # Extract DOE Settings
        wavelength = user_input.get('wavelength', 532e-9)
        pixel_size = user_input.get('pixel_size', 1e-6)
        device_diameter = user_input.get('device_diameter', 256e-6)
        propagation_type = user_input.get('propagation_type', 'fft')
        working_distance = user_input.get('working_distance')
        target_span = user_input.get('target_span')

        # Calculate DOE pixels from device diameter and pixel size
        doe_pixels = user_input.get('doe_pixels')
        if doe_pixels is None:
            doe_size = int(device_diameter / pixel_size)
            doe_pixels = [doe_size, doe_size]
        elif isinstance(doe_pixels, (int, float)):
            doe_pixels = [int(doe_pixels), int(doe_pixels)]

        # Get simulation parameters
        simulation_pixels = user_input.get('simulation_pixels')
        if simulation_pixels is None:
            simulation_pixels = list(target_tensor.shape[-2:])
        elif isinstance(simulation_pixels, (int, float)):
            simulation_pixels = [int(simulation_pixels), int(simulation_pixels)]

        period_pixels = user_input.get('period_pixels')
        if period_pixels is None:
            period_pixels = simulation_pixels[0]

        # Create physical constants
        physical = PhysicalConstants(
            wavelength=wavelength,
            pixel_size=pixel_size,
            refraction_index=1.62  # Default value
        )

        # Determine propagation type
        prop_type = PropagationType(propagation_type) if propagation_type else PropagationType.FFT

        # Get advanced settings (needed for target_margin in SFR)
        advanced = user_input.get('advanced', {})

        # Create appropriate structured params based on propagation type
        if prop_type == PropagationType.FFT:
            params = FFTParams(
                physical=physical,
                period_pixels=tuple(simulation_pixels),
                doe_total_pixels=tuple(doe_pixels)
            )
        elif prop_type == PropagationType.ASM:
            # Calculate target pixels for ASM
            target_px = list(simulation_pixels)
            if target_span and working_distance:
                # Calculate output pixels to cover target span
                output_size_px = int(target_span / pixel_size)
                target_px = [output_size_px, output_size_px]
            # Ensure target >= DOE for linear convolution
            target_px[0] = max(target_px[0], doe_pixels[0])
            target_px[1] = max(target_px[1], doe_pixels[1])

            params = ASMParams(
                physical=physical,
                doe_pixels=tuple(doe_pixels),
                working_distances=[working_distance] if working_distance else [0.01],
                target_pixels=tuple(target_px),
                upsample_factor=1
            )
        elif prop_type == PropagationType.SFR:
            # Calculate target size and resolution for SFR
            # IMPORTANT: target_resolution must match the target_pattern shape!
            # The target_pattern was already created by wizard with correct resolution.
            # target_size must include margin factor to match wizard coordinate system.
            target_margin = advanced.get('target_margin', 0.1)
            margin_factor = 1.0 + target_margin  # Default: 1.1

            # Use target_pattern shape as target_resolution (must match exactly!)
            target_res = tuple(target_tensor.shape[-2:])

            if target_span and working_distance:
                # Apply margin factor to target_size (must match wizard!)
                target_size_m = (target_span * margin_factor, target_span * margin_factor)
            else:
                # Fallback: use DOE size as target
                doe_size_m = doe_pixels[0] * pixel_size
                target_size_m = (doe_size_m, doe_size_m)

            params = SFRParams(
                physical=physical,
                doe_pixels=tuple(doe_pixels),
                working_distances=[working_distance] if working_distance else [0.01],
                target_size=target_size_m,
                target_resolution=target_res,
                upsample_factor=1
            )
        else:
            # Default to FFT
            params = FFTParams(
                physical=physical,
                period_pixels=tuple(simulation_pixels),
                doe_total_pixels=tuple(doe_pixels)
            )

        # Create propagator config from structured params
        # This ensures all propagation-specific parameters are included
        # (e.g., output_size, output_resolution for SFR)
        propagator_config = params.to_propagator_config()

        # Create optimization config from user input
        opt_input = user_input.get('optimization', {})
        loss_type_str = opt_input.get('loss_type', 'L2')
        try:
            loss_type = LossType(loss_type_str)
        except ValueError:
            loss_type = LossType.L2

        loss_config = LossConfig(loss_type=loss_type)

        phase_method_str = opt_input.get('phase_method', 'SGD')
        try:
            phase_method = OptMethod(phase_method_str)
        except ValueError:
            phase_method = OptMethod.SGD

        opt_config = OptimizationConfig(
            phase_method=phase_method,
            phase_lr=opt_input.get('phase_lr', 3e-9),
            phase_iters=opt_input.get('phase_iters', 1000),
            loss=loss_config,
            pixel_multiplier=opt_input.get('pixel_multiplier', 1),
            simulation_upsample=opt_input.get('simulation_upsample', 1),
        )

        # Metadata from advanced settings
        # Note: Efficiency calculation derives target_indices from target_pattern
        # No need for wizard-specific metadata
        advanced = user_input.get('advanced', {})
        metadata = {
            'target_margin': advanced.get('target_margin', 0.1),
            'progress_interval': advanced.get('progress_interval', 50),
        }

        return WizardOutput(
            structured_params=params,
            propagator_config=propagator_config,
            optimization_config=opt_config,
            target_pattern=target_tensor,
            computed_values={
                'period_pixels': period_pixels,
                'num_periods': [doe_pixels[0] // period_pixels, doe_pixels[1] // period_pixels] if period_pixels else [1, 1],
            },
            warnings=[],
            metadata=metadata
        )

    def _apply_doe_settings_overrides(
        self,
        wizard_output: WizardOutput,
        user_input: Dict[str, Any]
    ) -> WizardOutput:
        """Apply DOE Settings overrides from frontend to wizard output.

        The wizard generates default structured params from high-level user input.
        The frontend then allows users to edit these DOE Settings directly.
        This method applies any user-edited values as overrides.

        Args:
            wizard_output: Output from wizard with default values
            user_input: User input dict that may contain DOE Settings overrides

        Returns:
            Modified WizardOutput with overrides applied
        """
        from ..params.optimization import OptimizationConfig, OptMethod, LossType, LossConfig

        params = wizard_output.structured_params
        opt_config = wizard_output.optimization_config

        # Check for DOE Settings overrides in user_input
        # These are sent from frontend getOptimizationRequest()

        # Override propagation_type if specified
        if 'propagation_type' in user_input:
            new_prop_type = user_input['propagation_type']
            # Note: Changing propagation_type requires rebuilding propagator_config
            # This is complex, so we log but don't change the type here
            # The wizard already chose the type based on working_distance and target_spec
            pass

        # Override simulation resolution if specified (for FFT params)
        if 'period_pixels' in user_input and hasattr(params, 'period_pixels'):
            period_val = user_input['period_pixels']
            if isinstance(period_val, (int, float)):
                params.period_pixels = (int(period_val), int(period_val))
            elif isinstance(period_val, (list, tuple)) and len(period_val) >= 2:
                params.period_pixels = (int(period_val[0]), int(period_val[1]))

        if 'simulation_pixels' in user_input and hasattr(params, 'simulation_pixels'):
            sim_val = user_input['simulation_pixels']
            if isinstance(sim_val, (int, float)):
                params.simulation_pixels = (int(sim_val), int(sim_val))
            elif isinstance(sim_val, (list, tuple)) and len(sim_val) >= 2:
                params.simulation_pixels = (int(sim_val[0]), int(sim_val[1]))

        if 'doe_pixels' in user_input and hasattr(params, 'doe_total_pixels'):
            doe_val = user_input['doe_pixels']
            if isinstance(doe_val, (int, float)):
                params.doe_total_pixels = (int(doe_val), int(doe_val))
            elif isinstance(doe_val, (list, tuple)) and len(doe_val) >= 2:
                params.doe_total_pixels = (int(doe_val[0]), int(doe_val[1]))

        # Apply optimization settings from 'optimization' dict
        opt_input = user_input.get('optimization', {})
        if opt_input:
            # Create new optimization config with overrides
            loss_type_str = opt_input.get('loss_type', opt_config.loss.loss_type.value)
            try:
                loss_type = LossType(loss_type_str)
            except ValueError:
                loss_type = opt_config.loss.loss_type

            loss_config = LossConfig(
                loss_type=loss_type,
                roi_enabled=opt_input.get('roi_enabled', opt_config.loss.roi_enabled),
                focal_params=opt_input.get('focal_params', opt_config.loss.focal_params),
            )

            phase_method_str = opt_input.get('phase_method', opt_config.phase_method.value)
            try:
                phase_method = OptMethod(phase_method_str)
            except ValueError:
                phase_method = opt_config.phase_method

            # Create new config with overrides
            new_opt_config = OptimizationConfig(
                phase_method=phase_method,
                phase_lr=opt_input.get('phase_lr', opt_config.phase_lr),
                phase_iters=opt_input.get('phase_iters', opt_config.phase_iters),
                loss=loss_config,
                fab_enabled=opt_input.get('fab_enabled', opt_config.fab_enabled),
                fab_lr=opt_input.get('fab_lr', opt_config.fab_lr),
                fab_iters=opt_input.get('fab_iters', opt_config.fab_iters),
                pixel_multiplier=opt_input.get('pixel_multiplier', opt_config.pixel_multiplier),
                simulation_upsample=opt_input.get('simulation_upsample', opt_config.simulation_upsample),
            )
            wizard_output.optimization_config = new_opt_config

        # Apply advanced settings from 'advanced' dict
        advanced = user_input.get('advanced', {})
        if advanced:
            # target_margin is used for ASM/SFR target padding
            if 'target_margin' in advanced:
                wizard_output.metadata['target_margin'] = advanced['target_margin']
            # progress_interval is used for reporting frequency
            if 'progress_interval' in advanced:
                wizard_output.metadata['progress_interval'] = advanced['progress_interval']

        return wizard_output

    def _run_optimization(
        self,
        wizard_output: WizardOutput,
        progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
        cancellation_token: Optional[CancellationToken] = None
    ) -> Optional[OptimizationResult]:
        """Run the actual optimization.

        Args:
            wizard_output: Output from wizard
            progress_callback: Progress callback
            cancellation_token: Cancellation token

        Returns:
            OptimizationResult or None if cancelled
        """
        params = wizard_output.structured_params
        opt_config = wizard_output.optimization_config
        target = wizard_output.target_pattern

        # Move target to device
        target = target.to(device=self.device, dtype=self.dtype)

        # Debug: print shapes and config
        prop_config = wizard_output.propagator_config
        print(f"[Optimizer] prop_type: {prop_config.prop_type}")
        print(f"[Optimizer] target shape: {target.shape}")
        print(f"[Optimizer] feature_size: {prop_config.feature_size}")
        if prop_config.output_size:
            print(f"[Optimizer] output_size: {prop_config.output_size}")
        if prop_config.output_resolution:
            print(f"[Optimizer] output_resolution: {prop_config.output_resolution}")

        # Build propagator
        propagator_builder = PropagatorBuilder(
            wizard_output.propagator_config,
            device=self.device,
            dtype=self.dtype
        )
        propagator = propagator_builder.build()

        # Get optimization resolution
        if isinstance(params, FFTParams):
            opt_resolution = params.simulation_pixels
        elif isinstance(params, (ASMParams, SFRParams)):
            opt_resolution = params.simulation_pixels
        else:
            opt_resolution = (256, 256)

        # Get pixel_multiplier from optimization config (if specified)
        # This groups DOE pixels to reduce effective resolution
        pixel_multiplier = opt_config.pixel_multiplier if hasattr(opt_config, 'pixel_multiplier') else 1
        pixel_multiplier = max(1, pixel_multiplier)  # Ensure at least 1

        # If pixel_multiplier > 1, reduce optimization resolution and downsample target
        original_resolution = opt_resolution
        if pixel_multiplier > 1:
            # Use ceiling division to ensure expanded size >= original size
            # This way we can crop to exact original size after np.repeat
            import math
            opt_resolution = (
                max(1, math.ceil(opt_resolution[0] / pixel_multiplier)),
                max(1, math.ceil(opt_resolution[1] / pixel_multiplier))
            )

            # Downsample target to match reduced resolution
            # Use adaptive average pooling for clean downsampling
            target = torch.nn.functional.adaptive_avg_pool2d(
                target,
                output_size=opt_resolution
            )

            # Rebuild propagator with adjusted feature_size (effective pixel is larger)
            # This maintains the same k-space range but with coarser sampling
            from ..params.base import PropagatorConfig
            orig_config = wizard_output.propagator_config
            # Scale feature_size by pixel_multiplier
            new_feature_size = (
                orig_config.feature_size[0] * pixel_multiplier,
                orig_config.feature_size[1] * pixel_multiplier
            )
            adjusted_config = PropagatorConfig(
                prop_type=orig_config.prop_type,
                feature_size=new_feature_size,
                wavelength=orig_config.wavelength,
                working_distance=orig_config.working_distance,
                output_size=orig_config.output_size,
                output_resolution=opt_resolution if orig_config.output_resolution else None,
                num_channels=orig_config.num_channels,
                precompute_kernels=orig_config.precompute_kernels
            )
            propagator = PropagatorBuilder(
                adjusted_config,
                device=self.device,
                dtype=self.dtype
            ).build()

        # Create loss function
        loss_fn = self._create_loss_function(opt_config, wizard_output)

        # Create progress reporter
        reporter = ProgressReporter(
            callback=progress_callback,
            report_interval=100,
            cancellation_token=cancellation_token
        )

        # Initialize height with small random values to break symmetry
        # Zero initialization causes zero gradients due to abs() at purely real values
        height = torch.randn(
            1, 1, opt_resolution[0], opt_resolution[1],
            device=self.device, dtype=self.dtype
        ) * 1e-3
        height.requires_grad_(True)

        # Get physical parameters for phase conversion
        physical = params.physical
        wavelength = torch.tensor(
            [[[[physical.wavelength]]]],
            device=self.device, dtype=self.dtype
        )
        refraction_index = torch.tensor(
            [[[[physical.refraction_index]]]],
            device=self.device, dtype=self.dtype
        )

        # Run SGD optimization
        optimizer = torch.optim.Adam([height], lr=opt_config.phase_lr)
        num_iters = opt_config.phase_iters

        reporter.start_stage('phase', num_iters)

        best_height = height.clone()
        best_loss = float('inf')

        # Convert target amplitude to normalized intensity for optimization
        # Intensity-based comparison works better for phase optimization
        target_intensity = target ** 2
        target_intensity = target_intensity / (target_intensity.sum() + 1e-10)

        for i in range(num_iters):
            optimizer.zero_grad()

            # Forward model: height -> phase -> field -> propagate
            phase = self._height_to_phase(height, wavelength, refraction_index)
            field = torch.exp(1j * phase.to(torch.complex64))
            output_field = propagator(field)
            output_amp = output_field.abs()

            # Debug: print shapes on first iteration
            if i == 0:
                print(f"[Optimizer] field shape: {field.shape}")
                print(f"[Optimizer] output_field shape: {output_field.shape}")
                print(f"[Optimizer] target_intensity shape: {target_intensity.shape}")

            # Compute normalized intensity for loss
            output_intensity = output_amp ** 2
            output_intensity_norm = output_intensity / (output_intensity.sum() + 1e-10)

            # Debug: print loss on first and every 100th iteration
            if i == 0 or i == 100:
                print(f"[Optimizer] iter {i}: loss={loss_fn(output_intensity_norm, target_intensity).item():.6f}")

            # Compute loss in intensity space
            loss = loss_fn(output_intensity_norm, target_intensity)

            # Backward and step
            loss.backward()
            optimizer.step()

            loss_val = loss.item()

            if loss_val < best_loss:
                best_loss = loss_val
                best_height = height.clone()

            # Report progress
            if not reporter.report(i, loss_val):
                return None  # Cancelled

        # Final evaluation
        with torch.no_grad():
            phase = self._height_to_phase(best_height, wavelength, refraction_index)
            field = torch.exp(1j * phase.to(torch.complex64))
            output_field = propagator(field)
            output_amp = output_field.abs()

        # Convert to numpy (detach to handle requires_grad tensors)
        # Use squeeze(0).squeeze(0) to preserve 2D shape for 1D splitters
        height_np = best_height.detach().squeeze(0).squeeze(0).cpu().numpy()
        phase_np = phase.detach().squeeze(0).squeeze(0).cpu().numpy() % (2 * np.pi)
        target_np = target.detach().squeeze(0).squeeze(0).cpu().numpy() ** 2  # Intensity
        simulated_np = output_amp.detach().squeeze(0).squeeze(0).cpu().numpy() ** 2

        # Normalize both to unit total energy for fair comparison and display
        # This matches how target was originally normalized in wizard
        target_np = target_np / (target_np.sum() + 1e-10)
        simulated_np = simulated_np / (simulated_np.sum() + 1e-10)

        # Store reduced-resolution arrays for metrics calculation BEFORE expansion
        # These are at the actual optimization resolution
        target_for_metrics = target_np.copy()
        simulated_for_metrics = simulated_np.copy()
        reduced_resolution = opt_resolution  # Current resolution (may be reduced)

        # If pixel_multiplier > 1, expand arrays back to original resolution for display
        if pixel_multiplier > 1:
            height_np = np.repeat(np.repeat(height_np, pixel_multiplier, axis=0), pixel_multiplier, axis=1)
            phase_np = np.repeat(np.repeat(phase_np, pixel_multiplier, axis=0), pixel_multiplier, axis=1)
            target_np = np.repeat(np.repeat(target_np, pixel_multiplier, axis=0), pixel_multiplier, axis=1)
            simulated_np = np.repeat(np.repeat(simulated_np, pixel_multiplier, axis=0), pixel_multiplier, axis=1)
            # Crop to original resolution if needed (in case of rounding)
            height_np = height_np[:original_resolution[0], :original_resolution[1]]
            phase_np = phase_np[:original_resolution[0], :original_resolution[1]]
            target_np = target_np[:original_resolution[0], :original_resolution[1]]
            simulated_np = simulated_np[:original_resolution[0], :original_resolution[1]]

        return OptimizationResult(
            height=height_np,
            phase=phase_np,
            target_intensity=target_np,
            simulated_intensity=simulated_np,
            final_loss=best_loss,
            wizard_output=wizard_output,
            # Pass metrics arrays and resolution info for proper efficiency calculation
            simulated_for_metrics=simulated_for_metrics,
            target_for_metrics=target_for_metrics,
            reduced_resolution=reduced_resolution,
            original_resolution=original_resolution,
            pixel_multiplier=pixel_multiplier
        )

    def _height_to_phase(
        self,
        height: torch.Tensor,
        wavelength: torch.Tensor,
        refraction_index: torch.Tensor
    ) -> torch.Tensor:
        """Convert height to phase."""
        height_2pi = wavelength / (refraction_index - 1)
        phase = 2 * np.pi * height / height_2pi
        return phase

    def _create_loss_function(
        self,
        opt_config,
        wizard_output: WizardOutput
    ) -> BaseLoss:
        """Create loss function from config."""
        loss_config = opt_config.loss

        # Default to L2
        if loss_config.loss_type.value in ('L1', 'L2'):
            return create_loss(loss_config.loss_type.value)

        # For focal efficiency, get spot positions from metadata
        if loss_config.loss_type.value == 'focal_efficiency':
            focal_params = loss_config.focal_params or {}
            spot_positions = focal_params.get('spot_positions', [(128, 128)])
            airy_radius = focal_params.get('airy_radius', 3.0)
            return create_loss(
                'focal_efficiency',
                spot_positions=spot_positions,
                airy_radius=airy_radius
            )

        return L2Loss()

    def _create_result_data(
        self,
        result: OptimizationResult,
        wizard_output: WizardOutput
    ) -> DOEResultData:
        """Create API result data from optimization result."""
        # Calculate actual diffraction efficiency from simulated intensity
        metrics = self._compute_metrics(result, wizard_output)

        params = wizard_output.structured_params
        metadata = wizard_output.metadata or {}
        strategy = metadata.get('strategy')

        # Build kwargs for optional fields
        kwargs = {
            'computed_params': wizard_output.computed_values,
            'splitter_info': metadata if 'grid_mode' in metadata else None,
        }

        # For FFT-based methods, include period_pixels for frontend tiling
        if isinstance(params, FFTParams):
            kwargs['period_pixels'] = params.period_pixels[0]  # Assuming square

            # For Strategy 2 (periodic_fresnel), compute combined phase with Fresnel
            if strategy == 'periodic_fresnel':
                period_h, period_w = params.period_pixels
                doe_h, doe_w = params.doe_total_pixels
                num_tiles_y = doe_h // period_h
                num_tiles_x = doe_w // period_w

                # Tile period phase to full device
                # Note: tiled size may be smaller than doe_total_pixels if not evenly divisible
                device_phase_np = np.tile(result.phase, (num_tiles_y, num_tiles_x))
                tiled_h, tiled_w = device_phase_np.shape

                # Add Fresnel lens phase
                working_distance = metadata.get('working_distance')
                wavelength = params.physical.wavelength
                pixel_size = params.physical.pixel_size

                if working_distance:
                    # Compute Fresnel phase with same shape as tiled device phase
                    fresnel_phase = self._compute_fresnel_phase(
                        shape=(tiled_h, tiled_w),
                        pixel_size=pixel_size,
                        wavelength=wavelength,
                        focal_length=working_distance
                    )
                    combined_phase = (device_phase_np + fresnel_phase) % (2 * np.pi)
                    kwargs['device_phase_with_fresnel'] = combined_phase
                    kwargs['fresnel_phase'] = fresnel_phase
                    kwargs['device_phase'] = device_phase_np

        return DOEResultData.from_arrays(
            height=result.height,
            phase=result.phase,
            target_intensity=result.target_intensity,
            simulated_intensity=result.simulated_intensity,
            metrics=metrics,
            **kwargs
        )

    def _compute_metrics(
        self,
        result: OptimizationResult,
        wizard_output: WizardOutput
    ) -> MetricsData:
        """Compute diffraction efficiency metrics from simulated intensity.

        Calculates:
        - Total efficiency: sum of energy in working orders / total energy
        - Order efficiencies: individual order efficiencies
        - Uniformity: 1 - (max - min) / (max + min)
        - Mean/std efficiency

        For ASM (physical space), integrates over Airy disk area.
        For FFT (k-space), uses single pixel values.
        """
        metadata = wizard_output.metadata or {}
        params = wizard_output.structured_params

        # Use reduced-resolution arrays for metrics if pixel_multiplier > 1
        # This ensures order positions align correctly
        pixel_multiplier = result.pixel_multiplier
        if pixel_multiplier > 1 and result.simulated_for_metrics is not None:
            simulated = result.simulated_for_metrics
            original_resolution = result.original_resolution
            reduced_resolution = result.reduced_resolution
        else:
            simulated = result.simulated_intensity
            original_resolution = None
            reduced_resolution = None

        # Get target indices from target pattern (non-zero positions)
        # This is DOE-type agnostic - works for splitter, diffuser, etc.
        target = result.target_for_metrics if result.target_for_metrics is not None else result.target_intensity
        target_indices = self._extract_target_indices(target)

        if not target_indices:
            # No target points, use loss-based estimate
            return MetricsData(total_efficiency=1.0 - result.final_loss)

        # Determine integration mode from propagation type (not splitter-specific)
        prop_type = wizard_output.propagator_config.prop_type.value if hasattr(wizard_output.propagator_config.prop_type, 'value') else str(wizard_output.propagator_config.prop_type)
        use_disk_integration = prop_type in ('asm', 'sfr')

        # Scale target indices if using reduced resolution
        if pixel_multiplier > 1 and original_resolution and reduced_resolution:
            # Scale factor: reduced / original
            scale_y = reduced_resolution[0] / original_resolution[0]
            scale_x = reduced_resolution[1] / original_resolution[1]
            # Scale positions and deduplicate to avoid counting same pixel multiple times
            scaled_set = set()
            for py, px in target_indices:
                scaled_pos = (int(round(py * scale_y)), int(round(px * scale_x)))
                scaled_set.add(scaled_pos)
            scaled_positions = list(scaled_set)
        else:
            scaled_positions = target_indices

        total_intensity = simulated.sum()
        if total_intensity < 1e-10:
            return MetricsData(total_efficiency=0.0)

        # Compute efficiency for each target spot
        # Handle both 1D (H, 1) and 2D (H, W) cases
        spot_efficiencies = []
        h = simulated.shape[0]
        w = simulated.shape[1] if len(simulated.shape) > 1 else 1

        # For ASM/SFR (physical space), integrate over Airy disk
        # For FFT (k-space), use single pixel values
        if use_disk_integration:
            # Calculate Airy disk radius for physical space integration
            airy_radius_pixels = self._compute_airy_radius(params)
            airy_radius_pixels = max(3, airy_radius_pixels)  # Minimum 3 pixels
            # Scale Airy radius if using reduced resolution
            if pixel_multiplier > 1:
                airy_radius_pixels = max(1, airy_radius_pixels // pixel_multiplier)

            for py, px in scaled_positions:
                eff = self._integrate_over_disk(
                    simulated, py, px, airy_radius_pixels, total_intensity
                )
                spot_efficiencies.append(eff)
        else:
            # FFT (k-space): single pixel values
            for py, px in scaled_positions:
                if 0 <= py < h and 0 <= px < w:
                    if len(simulated.shape) > 1:
                        eff = float(simulated[py, px]) / float(total_intensity)
                    else:
                        eff = float(simulated[py]) / float(total_intensity)
                    spot_efficiencies.append(eff)
                else:
                    spot_efficiencies.append(0.0)

        # Aggregate metrics
        total_efficiency = sum(spot_efficiencies)
        mean_efficiency = np.mean(spot_efficiencies) if spot_efficiencies else 0.0
        std_efficiency = np.std(spot_efficiencies) if spot_efficiencies else 0.0

        # Uniformity: 1 - (max - min) / (max + min)
        if spot_efficiencies and max(spot_efficiencies) + min(spot_efficiencies) > 1e-10:
            uniformity = 1.0 - (max(spot_efficiencies) - min(spot_efficiencies)) / (
                max(spot_efficiencies) + min(spot_efficiencies)
            )
        else:
            uniformity = 0.0

        return MetricsData(
            total_efficiency=total_efficiency,
            uniformity=uniformity,
            mean_efficiency=mean_efficiency,
            std_efficiency=std_efficiency,
            order_efficiencies=spot_efficiencies,  # Keep field name for API compatibility
        )

    def _extract_target_indices(self, target: np.ndarray, threshold: float = 1e-6) -> List[Tuple[int, int]]:
        """Extract target point indices from target pattern.

        Finds all positions where target intensity is above threshold.
        This is DOE-type agnostic - works for any target pattern.

        Args:
            target: Target intensity array (1D or 2D)
            threshold: Minimum intensity to consider as target point

        Returns:
            List of (py, px) index tuples
        """
        if target is None:
            return []

        # Handle 1D case - return as (row=0, col=position)
        if target.ndim == 1 or (target.ndim == 2 and min(target.shape) == 1):
            target_1d = target.reshape(-1)
            indices = np.where(target_1d > threshold)[0]
            return [(0, int(i)) for i in indices]

        # 2D case
        positions = np.where(target > threshold)
        return [(int(py), int(px)) for py, px in zip(positions[0], positions[1])]

    def _compute_airy_radius(self, params) -> int:
        """Compute Airy disk radius in pixels for ASM/SFR."""
        if isinstance(params, ASMParams):
            wavelength = params.physical.wavelength
            pixel_size = params.physical.pixel_size
            doe_size = params.doe_pixels[0] * pixel_size
            working_distance = params.working_distances[0]
            airy_radius_m = 1.22 * wavelength * working_distance / doe_size
            return int(airy_radius_m / pixel_size)
        elif isinstance(params, SFRParams):
            wavelength = params.physical.wavelength
            pixel_size_target = params.target_pixel_size[0]
            doe_size = params.doe_pixels[0] * params.physical.pixel_size
            working_distance = params.working_distances[0]
            airy_radius_m = 1.22 * wavelength * working_distance / doe_size
            return int(airy_radius_m / pixel_size_target)
        return 3  # Default minimum

    def _integrate_over_disk(
        self,
        intensity: np.ndarray,
        cy: int,
        cx: int,
        radius: int,
        total_intensity: float
    ) -> float:
        """Integrate intensity over a circular disk.

        Args:
            intensity: 2D intensity array
            cy, cx: Center position
            radius: Disk radius in pixels
            total_intensity: Total intensity for normalization

        Returns:
            Efficiency (integrated intensity / total)
        """
        h = intensity.shape[0]
        w = intensity.shape[1] if len(intensity.shape) > 1 else 1

        if len(intensity.shape) == 1:
            # 1D case: sum over line segment
            start = max(0, cy - radius)
            end = min(h, cy + radius + 1)
            return float(intensity[start:end].sum()) / float(total_intensity)

        # 2D case: sum over circular disk
        y_min = max(0, cy - radius)
        y_max = min(h, cy + radius + 1)
        x_min = max(0, cx - radius)
        x_max = min(w, cx + radius + 1)

        disk_sum = 0.0
        for y in range(y_min, y_max):
            for x in range(x_min, x_max):
                if (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2:
                    disk_sum += float(intensity[y, x])

        return disk_sum / float(total_intensity)

    def _create_visualization_data(
        self,
        result: OptimizationResult,
        wizard_output: WizardOutput
    ) -> VisualizationData:
        """Create visualization data for Plotly."""
        params = wizard_output.structured_params
        metadata = wizard_output.metadata or {}
        pixel_size = params.physical.pixel_size
        strategy = metadata.get('strategy')

        # Create basic phase and intensity visualizations
        period_phase = create_phase_heatmap(
            result.phase,
            title="Optimized Phase (Period)",
            pixel_size=pixel_size,
            unit="um"
        )

        target_intensity = create_intensity_heatmap(
            result.target_intensity,
            title="Target Intensity"
        )

        simulated_intensity = create_intensity_heatmap(
            result.simulated_intensity,
            title="Simulated Intensity"
        )

        # For FFT-based methods (including Strategy 2), create full device phase
        device_phase = None
        device_phase_with_fresnel = None
        finite_distance_eval = None

        if isinstance(params, FFTParams):
            # Tile period phase to full device
            period_h, period_w = params.period_pixels
            doe_h, doe_w = params.doe_total_pixels

            num_tiles_y = doe_h // period_h
            num_tiles_x = doe_w // period_w

            if num_tiles_y > 1 or num_tiles_x > 1:
                device_phase_np = np.tile(result.phase, (num_tiles_y, num_tiles_x))
                device_phase = create_phase_heatmap(
                    device_phase_np,
                    title="Full Device Phase",
                    pixel_size=pixel_size,
                    unit="um"
                )

                # For Strategy 2 (periodic_fresnel), add Fresnel lens overlay
                if strategy == 'periodic_fresnel':
                    working_distance = metadata.get('working_distance')
                    wavelength = params.physical.wavelength

                    if working_distance:
                        # Compute Fresnel lens phase with same shape as tiled device phase
                        tiled_h, tiled_w = device_phase_np.shape
                        fresnel_phase = self._compute_fresnel_phase(
                            shape=(tiled_h, tiled_w),
                            pixel_size=pixel_size,
                            wavelength=wavelength,
                            focal_length=working_distance
                        )

                        combined_phase = (device_phase_np + fresnel_phase) % (2 * np.pi)

                        device_phase_with_fresnel = create_phase_heatmap(
                            combined_phase,
                            title="Device Phase + Fresnel Lens",
                            pixel_size=pixel_size,
                            unit="um"
                        )

                        # Run finite distance evaluation using SFR
                        target_span = metadata.get('target_span')
                        order_angles = metadata.get('order_angles', [])
                        working_orders = metadata.get('working_orders', [])

                        if target_span and order_angles:
                            try:
                                from ..evaluation.evaluator import evaluate_finite_distance_splitter
                                eval_result = evaluate_finite_distance_splitter(
                                    device_phase=device_phase_np,
                                    wavelength=wavelength,
                                    pixel_size=pixel_size,
                                    working_distance=working_distance,
                                    target_span=target_span,
                                    working_orders=working_orders,
                                    order_angles=order_angles,
                                    fresnel_phase=fresnel_phase,
                                    output_resolution=(512, 512),
                                    device=self.device,
                                    dtype=self.dtype
                                )
                                # Create finite distance intensity visualization
                                finite_distance_eval = create_intensity_heatmap(
                                    eval_result.simulated_intensity,
                                    title=f"SFR Propagation (z={working_distance*1e3:.1f}mm)",
                                    log_scale=True
                                )
                            except Exception as e:
                                print(f"Warning: Finite distance evaluation failed: {e}")

        return VisualizationData(
            period_phase=period_phase,
            device_phase=device_phase,
            device_phase_with_fresnel=device_phase_with_fresnel,
            target_intensity=target_intensity,
            simulated_intensity=simulated_intensity,
            finite_distance_intensity=finite_distance_eval,
            summary={
                'final_loss': result.final_loss,
                'strategy': strategy,
                **wizard_output.computed_values
            }
        )

    def _compute_fresnel_phase(
        self,
        shape: tuple,
        pixel_size: float,
        wavelength: float,
        focal_length: float
    ) -> np.ndarray:
        """Compute Fresnel lens phase for focusing at given distance."""
        h, w = shape
        y = (np.arange(h) - h / 2) * pixel_size
        x = (np.arange(w) - w / 2) * pixel_size
        YY, XX = np.meshgrid(y, x, indexing='ij')

        k = 2 * np.pi / wavelength
        phase = -k / (2 * focal_length) * (XX ** 2 + YY ** 2)
        phase = phase % (2 * np.pi)

        return phase


def run_optimization(
    user_input: Dict[str, Any],
    progress_callback: Optional[Callable[[ProgressInfo], None]] = None,
    cancellation_token: Optional[CancellationToken] = None,
    max_resolution: int = 2000
) -> OptimizationResponse:
    """Convenience function to run optimization from user input.

    Args:
        user_input: User input dictionary
        progress_callback: Optional progress callback
        cancellation_token: Optional cancellation token
        max_resolution: Maximum simulation resolution

    Returns:
        OptimizationResponse
    """
    request = OptimizationRequest.from_json(user_input)
    runner = OptimizationRunner(max_resolution=max_resolution)
    return runner.run(request, progress_callback, cancellation_token)
