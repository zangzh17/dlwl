"""
Structured error and warning types for validation feedback.

This module provides standardized error/warning types that can be
serialized to JSON for frontend display.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from enum import Enum


class ErrorCode(Enum):
    """Error codes for validation failures."""
    # Parameter errors
    MISSING_REQUIRED = "MISSING_REQUIRED"
    INVALID_TYPE = "INVALID_TYPE"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    INVALID_COMBINATION = "INVALID_COMBINATION"

    # Physical constraint errors
    RESOLUTION_EXCEEDED = "RESOLUTION_EXCEEDED"
    TARGET_RESOLUTION_EXCEEDED = "TARGET_RESOLUTION_EXCEEDED"
    SAMPLING_VIOLATED = "SAMPLING_VIOLATED"
    DIFFRACTION_LIMIT = "DIFFRACTION_LIMIT"
    ANGLE_OUT_OF_RANGE = "ANGLE_OUT_OF_RANGE"

    # Resource errors
    MEMORY_EXCEEDED = "MEMORY_EXCEEDED"
    COMPUTATION_LIMIT = "COMPUTATION_LIMIT"

    # File errors
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    INVALID_IMAGE = "INVALID_IMAGE"

    # Fabrication errors
    FAB_RECIPE_NOT_FOUND = "FAB_RECIPE_NOT_FOUND"
    FAB_INCOMPATIBLE = "FAB_INCOMPATIBLE"


class WarningCode(Enum):
    """Warning codes for non-fatal issues."""
    # Performance warnings
    LARGE_COMPUTATION = "LARGE_COMPUTATION"
    SUBOPTIMAL_PARAMS = "SUBOPTIMAL_PARAMS"

    # Physical warnings
    LOW_EFFICIENCY_EXPECTED = "LOW_EFFICIENCY_EXPECTED"
    TOLERANCE_TIGHT = "TOLERANCE_TIGHT"
    ALIASING_RISK = "ALIASING_RISK"

    # Parameter suggestions
    CONSIDER_ALTERNATIVE = "CONSIDER_ALTERNATIVE"
    DEFAULT_USED = "DEFAULT_USED"


@dataclass
class ValidationError:
    """Structured validation error.

    Attributes:
        code: Error code enum value
        message: Human-readable error message
        field: Parameter field that caused the error (if applicable)
        details: Additional error details
        suggestion: Suggested fix
    """
    code: ErrorCode
    message: str
    field: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'code': self.code.value,
            'message': self.message,
            'severity': 'error',
        }
        if self.field:
            result['field'] = self.field
        if self.details:
            result['details'] = self.details
        if self.suggestion:
            result['suggestion'] = self.suggestion
        return result


@dataclass
class ValidationWarning:
    """Structured validation warning.

    Attributes:
        code: Warning code enum value
        message: Human-readable warning message
        field: Parameter field that triggered the warning (if applicable)
        suggestion: Suggested improvement
    """
    code: WarningCode
    message: str
    field: Optional[str] = None
    suggestion: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'code': self.code.value,
            'message': self.message,
            'severity': 'warning',
        }
        if self.field:
            result['field'] = self.field
        if self.suggestion:
            result['suggestion'] = self.suggestion
        return result


@dataclass
class ValidationResult:
    """Complete validation result.

    Attributes:
        is_valid: Whether validation passed (no errors)
        errors: List of validation errors
        warnings: List of validation warnings
    """
    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationWarning] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'is_valid': self.is_valid,
            'errors': [e.to_dict() for e in self.errors],
            'warnings': [w.to_dict() for w in self.warnings],
        }

    def add_error(
        self,
        code: ErrorCode,
        message: str,
        field: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        suggestion: Optional[str] = None
    ) -> None:
        """Add an error to the result."""
        self.errors.append(ValidationError(
            code=code,
            message=message,
            field=field,
            details=details,
            suggestion=suggestion
        ))
        self.is_valid = False

    def add_warning(
        self,
        code: WarningCode,
        message: str,
        field: Optional[str] = None,
        suggestion: Optional[str] = None
    ) -> None:
        """Add a warning to the result."""
        self.warnings.append(ValidationWarning(
            code=code,
            message=message,
            field=field,
            suggestion=suggestion
        ))

    @classmethod
    def success(cls) -> 'ValidationResult':
        """Create a successful validation result."""
        return cls(is_valid=True)

    @classmethod
    def failure(cls, error: ValidationError) -> 'ValidationResult':
        """Create a failed validation result with a single error."""
        return cls(is_valid=False, errors=[error])
