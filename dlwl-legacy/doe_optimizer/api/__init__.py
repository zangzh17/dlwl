"""
API layer for DOE optimizer frontend communication.

This module provides JSON-serializable data structures for:
- User input schemas (request)
- Optimization results (response)
- Validation errors and warnings
"""

from .schemas import (
    DOETypeSchema,
    SplitterModeSchema,
    TargetTypeSchema,
    DeviceShapeSchema,
    UserInputSchema,
    SplitterTargetSpec,
    DiffuserTargetSpec,
    LensTargetSpec,
    CustomTargetSpec,
)
from .request import OptimizationRequest
from .response import OptimizationResponse, DOEResultData
from .errors import (
    ValidationError,
    ValidationWarning,
    ErrorCode,
    WarningCode,
)

__all__ = [
    # Schemas
    'DOETypeSchema',
    'SplitterModeSchema',
    'TargetTypeSchema',
    'DeviceShapeSchema',
    'UserInputSchema',
    'SplitterTargetSpec',
    'DiffuserTargetSpec',
    'LensTargetSpec',
    'CustomTargetSpec',
    # Request/Response
    'OptimizationRequest',
    'OptimizationResponse',
    'DOEResultData',
    # Errors
    'ValidationError',
    'ValidationWarning',
    'ErrorCode',
    'WarningCode',
]
