"""
Validation API Routes.

Handles parameter validation with structured error messages.
"""

from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

from doe_optimizer import generate_params, validate_params


router = APIRouter(prefix="/api", tags=["validation"])


class ValidationRequest(BaseModel):
    """Request schema for parameter validation."""
    doe_type: str = Field(..., description="DOE type")
    wavelength: float = Field(532e-9, gt=0)
    device_diameter: float = Field(256e-6, gt=0)
    pixel_size: float = Field(1e-6, gt=0)
    device_shape: str = Field("square")
    target_spec: Dict[str, Any] = Field(...)
    working_distance: Optional[float] = Field(None)
    refraction_index: float = Field(1.62)


class ValidationMessage(BaseModel):
    """Validation message (error or warning)."""
    code: str
    message: str
    field: Optional[str] = None
    suggestion: Optional[str] = None
    severity: str = "error"


class ValidationResponse(BaseModel):
    """Response schema for validation."""
    is_valid: bool
    errors: List[ValidationMessage] = []
    warnings: List[ValidationMessage] = []


@router.post("/validate", response_model=ValidationResponse)
async def validate_user_input(request: ValidationRequest) -> ValidationResponse:
    """Validate user input parameters.

    Returns errors that prevent optimization and warnings for potential issues.
    """
    errors = []
    warnings = []

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

        # Basic input validation
        if request.wavelength <= 0:
            errors.append(ValidationMessage(
                code="INVALID_WAVELENGTH",
                message="Wavelength must be positive",
                field="wavelength"
            ))

        if request.pixel_size <= 0:
            errors.append(ValidationMessage(
                code="INVALID_PIXEL_SIZE",
                message="Pixel size must be positive",
                field="pixel_size"
            ))

        if request.device_diameter <= 0:
            errors.append(ValidationMessage(
                code="INVALID_DEVICE_DIAMETER",
                message="Device diameter must be positive",
                field="device_diameter"
            ))

        if request.pixel_size > request.device_diameter:
            errors.append(ValidationMessage(
                code="PIXEL_SIZE_TOO_LARGE",
                message="Pixel size cannot exceed device diameter",
                field="pixel_size"
            ))

        # Check target_spec for target_type='size' requirements
        target_spec = request.target_spec
        if target_spec.get('target_type') == 'size':
            if request.working_distance is None:
                errors.append(ValidationMessage(
                    code="MISSING_WORKING_DISTANCE",
                    message="Working distance is required when target_type is 'size'",
                    field="working_distance"
                ))

        if request.working_distance is not None and request.working_distance < 0:
            errors.append(ValidationMessage(
                code="INVALID_WORKING_DISTANCE",
                message="Working distance cannot be negative",
                field="working_distance"
            ))

        # Splitter-specific validation
        if 'splitter' in request.doe_type:
            num_spots = target_spec.get('num_spots')
            if num_spots is not None:
                if isinstance(num_spots, int):
                    if num_spots <= 0:
                        errors.append(ValidationMessage(
                            code="INVALID_NUM_SPOTS",
                            message="Number of spots must be positive",
                            field="target_spec.num_spots"
                        ))
                elif isinstance(num_spots, list):
                    if any(n <= 0 for n in num_spots):
                        errors.append(ValidationMessage(
                            code="INVALID_NUM_SPOTS",
                            message="All spot counts must be positive",
                            field="target_spec.num_spots"
                        ))

            target_span = target_spec.get('target_span')
            if target_span is not None:
                spans = [target_span] if isinstance(target_span, (int, float)) else target_span
                if any(s <= 0 for s in spans):
                    errors.append(ValidationMessage(
                        code="INVALID_TARGET_SPAN",
                        message="Target span must be positive",
                        field="target_spec.target_span"
                    ))

        # If basic validation passes, try generating params for deeper validation
        if not errors:
            try:
                wizard_output = generate_params(user_input)

                # Extract warnings from wizard output
                for w in wizard_output.warnings:
                    if hasattr(w, 'message'):
                        warnings.append(ValidationMessage(
                            code=str(getattr(w, 'code', 'WARNING')),
                            message=w.message,
                            field=getattr(w, 'field', None),
                            severity="warning"
                        ))
                    else:
                        warnings.append(ValidationMessage(
                            code="WARNING",
                            message=str(w),
                            severity="warning"
                        ))

                # Run structured params validation
                validation_result = validate_params(wizard_output.structured_params)

                if not validation_result.is_valid:
                    for msg in validation_result.errors:
                        errors.append(ValidationMessage(
                            code=str(msg.code),
                            message=msg.message,
                            field=msg.field,
                            suggestion=msg.suggestion,
                            severity="error"
                        ))

                for msg in validation_result.warnings:
                    warnings.append(ValidationMessage(
                        code=str(msg.code),
                        message=msg.message,
                        field=msg.field,
                        suggestion=msg.suggestion,
                        severity="warning"
                    ))

            except Exception as e:
                errors.append(ValidationMessage(
                    code="PARAM_GENERATION_ERROR",
                    message=f"Failed to generate parameters: {str(e)}",
                    severity="error"
                ))

        # Sampling theorem warning
        if request.pixel_size > request.wavelength / 2:
            warnings.append(ValidationMessage(
                code="SAMPLING_WARNING",
                message="Pixel size exceeds wavelength/2, which may cause aliasing",
                field="pixel_size",
                suggestion="Consider using smaller pixel size for accurate simulation",
                severity="warning"
            ))

    except Exception as e:
        errors.append(ValidationMessage(
            code="VALIDATION_ERROR",
            message=str(e),
            severity="error"
        ))

    return ValidationResponse(
        is_valid=len(errors) == 0,
        errors=errors,
        warnings=warnings
    )
