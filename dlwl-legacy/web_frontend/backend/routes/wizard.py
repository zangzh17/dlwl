"""
Wizard API Routes.

Handles parameter generation from user-friendly wizard input.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union

from doe_optimizer import generate_params, WizardOutput
from doe_optimizer.params.base import get_prop_type


router = APIRouter(prefix="/api", tags=["wizard"])


class WizardRequest(BaseModel):
    """Request schema for wizard parameter generation."""
    doe_type: str = Field(..., description="DOE type: splitter_1d, splitter_2d, diffuser, lens, custom")
    wavelength: float = Field(532e-9, gt=0, description="Wavelength in meters")
    device_diameter: float = Field(256e-6, gt=0, description="Device diameter in meters")
    pixel_size: float = Field(1e-6, gt=0, description="Pixel size in meters")
    device_shape: str = Field("square", pattern="^(square|circular)$")
    target_spec: Dict[str, Any] = Field(..., description="Target specification")
    working_distance: Optional[float] = Field(None, ge=0, description="Working distance in meters")
    refraction_index: float = Field(1.62, gt=1, description="Material refractive index")

    # Optional optimization settings
    optimization: Optional[Dict[str, Any]] = Field(None, description="Optimization settings override")
    # Optional advanced settings (includes target_margin)
    advanced: Optional[Dict[str, Any]] = Field(None, description="Advanced settings (target_margin, etc.)")


class WizardResponse(BaseModel):
    """Response schema for wizard parameter generation.

    Returns target_pattern so it can be stored in DOE Settings.
    Preview/Optimization will use this stored pattern, not call wizard again.
    """
    success: bool
    structured_params: Optional[Dict[str, Any]] = None
    propagator_config: Optional[Dict[str, Any]] = None
    computed_values: Optional[Dict[str, Any]] = None
    warnings: List[Dict[str, Any]] = []
    metadata: Optional[Dict[str, Any]] = None
    # Target pattern as 2D list (stored in DOE Settings, used by Preview/Optimization)
    target_pattern: Optional[List[List[float]]] = None
    error: Optional[str] = None


def _make_serializable(obj):
    """Recursively convert numpy types to Python native types for JSON serialization."""
    import numpy as np

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    else:
        return obj


def wizard_output_to_dict(output: WizardOutput) -> Dict[str, Any]:
    """Convert WizardOutput to serializable dictionary."""
    # Get structured params dict
    params = output.structured_params
    prop_type = get_prop_type(params)

    params_dict = {
        'propagation_type': prop_type.value,
        'physical': {
            'wavelength': params.physical.wavelength,
            'pixel_size': params.physical.pixel_size,
            'refraction_index': params.physical.refraction_index,
        }
    }

    # Add type-specific params
    if hasattr(params, 'doe_pixels'):
        params_dict['doe_pixels'] = list(params.doe_pixels)

    # simulation_pixels meaning varies by propagation type:
    # - FFT: period_pixels (optimization unit)
    # - ASM: output_pixels (target face resolution)
    # - SFR: doe_pixels (DOE resolution, zoom-FFT handles target)
    if prop_type.value == 'asm' and hasattr(params, 'output_pixels'):
        # For ASM, simulation_pixels represents target face pixels
        params_dict['simulation_pixels'] = list(params.output_pixels)
    elif hasattr(params, 'simulation_pixels'):
        params_dict['simulation_pixels'] = list(params.simulation_pixels)

    if hasattr(params, 'working_distances'):
        params_dict['working_distances'] = list(params.working_distances)
    if hasattr(params, 'target_pixel_size'):
        params_dict['target_pixel_size'] = list(params.target_pixel_size)
    if hasattr(params, 'target_pixels'):
        params_dict['target_pixels'] = list(params.target_pixels)

    # Propagator config
    prop_config = output.propagator_config
    prop_config_dict = {
        'propagation_type': prop_config.prop_type.value if hasattr(prop_config.prop_type, 'value') else str(prop_config.prop_type),
    }
    if hasattr(prop_config, 'working_distance'):
        prop_config_dict['working_distance'] = prop_config.working_distance

    # Warnings
    warnings = []
    for w in output.warnings:
        if hasattr(w, 'to_dict'):
            warnings.append(w.to_dict())
        elif hasattr(w, '__dict__'):
            warnings.append({
                'code': str(getattr(w, 'code', 'UNKNOWN')),
                'message': getattr(w, 'message', str(w)),
                'field': getattr(w, 'field', None),
            })
        else:
            warnings.append({'message': str(w)})

    return {
        'structured_params': _make_serializable(params_dict),
        'propagator_config': _make_serializable(prop_config_dict),
        'computed_values': _make_serializable(output.computed_values or {}),
        'warnings': warnings,
        'metadata': _make_serializable(output.metadata or {}),
    }


def _reset_cuda():
    """Reset CUDA state aggressively after error."""
    import torch
    import gc
    try:
        # Force garbage collection first
        gc.collect()

        if torch.cuda.is_available():
            # Clear all cached memory
            torch.cuda.empty_cache()

            # Reset peak memory stats
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass

            # Synchronize to ensure all operations complete
            try:
                torch.cuda.synchronize()
            except Exception:
                pass

            # Reset accumulated memory stats
            try:
                torch.cuda.reset_accumulated_memory_stats()
            except Exception:
                pass
    except Exception:
        pass


def _generate_params_with_cuda_recovery(user_input: dict):
    """Generate params with CUDA error recovery."""
    import torch
    import gc
    import os

    # Reset CUDA before attempting (clears any prior bad state)
    _reset_cuda()
    gc.collect()

    # Force CPU mode to avoid CUDA issues
    # This ensures reliability at the cost of GPU acceleration for preview
    os.environ['FORCE_CPU'] = '1'

    try:
        return generate_params(user_input)
    except RuntimeError as e:
        error_str = str(e)
        print(f"[wizard] Error during generate_params: {error_str[:200]}")

        if 'CUDA' in error_str or 'cuda' in error_str:
            # CUDA error - aggressive reset and retry on CPU
            print("[wizard] CUDA error detected, forcing CPU mode")
            _reset_cuda()
            gc.collect()

            # Force CPU by setting device in wizard
            import doe_optimizer.wizard.base as wizard_base
            original_get_device = wizard_base.BaseWizard._get_device

            def force_cpu(self):
                return torch.device('cpu')

            wizard_base.BaseWizard._get_device = force_cpu
            try:
                result = generate_params(user_input)
                print("[wizard] CPU fallback successful")
                return result
            finally:
                wizard_base.BaseWizard._get_device = original_get_device
                _reset_cuda()
        else:
            raise
    finally:
        # Clear the force CPU flag after this request
        if 'FORCE_CPU' in os.environ:
            del os.environ['FORCE_CPU']


@router.post("/wizard", response_model=WizardResponse)
async def generate_wizard_params(request: WizardRequest) -> WizardResponse:
    """Generate structured parameters from wizard input.

    Takes user-friendly parameters and converts them to structured
    optimization parameters.
    """
    try:
        # Build user input dict
        user_input = {
            'doe_type': request.doe_type,
            'wavelength': request.wavelength,
            'device_diameter': request.device_diameter,
            'pixel_size': request.pixel_size,
            'device_shape': request.device_shape,
            'target_spec': request.target_spec,
            'refraction_index': request.refraction_index,
        }

        if request.working_distance is not None:
            user_input['working_distance'] = request.working_distance

        if request.optimization:
            user_input['optimization'] = request.optimization

        if request.advanced:
            user_input['advanced'] = request.advanced
            print(f"[wizard] Advanced settings: {request.advanced}")

        print(f"[wizard] User input keys: {list(user_input.keys())}")

        # Generate parameters via wizard with CUDA error recovery
        wizard_output = _generate_params_with_cuda_recovery(user_input)

        # Convert to response format
        result = wizard_output_to_dict(wizard_output)

        # Extract target pattern as 2D list for storage in frontend
        target_pattern = None
        if wizard_output.target_pattern is not None:
            target_np = wizard_output.target_pattern.squeeze().cpu().numpy()
            target_pattern = target_np.tolist()

        return WizardResponse(
            success=True,
            structured_params=result['structured_params'],
            propagator_config=result['propagator_config'],
            computed_values=result['computed_values'],
            warnings=result['warnings'],
            metadata=result['metadata'],
            target_pattern=target_pattern,
        )

    except Exception as e:
        # Try to reset CUDA even on other errors
        _reset_cuda()
        return WizardResponse(
            success=False,
            error=str(e)
        )
