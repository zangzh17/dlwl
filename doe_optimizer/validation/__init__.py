"""
Validation layer for structured parameters.

This module provides:
- Structured validation results with error/warning messages
- Physical constraint validation
- Computational limit checking
"""

from .messages import (
    Severity,
    ValidationMessage,
    ValidationResult,
)
from .validator import (
    StructuredParamsValidator,
    validate_params,
)

__all__ = [
    'Severity',
    'ValidationMessage',
    'ValidationResult',
    'StructuredParamsValidator',
    'validate_params',
]
