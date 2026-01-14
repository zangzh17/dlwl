"""
Optimization request data structures.

This module defines the request format for DOE optimization.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, TYPE_CHECKING
import uuid

from .schemas import UserInputSchema

if TYPE_CHECKING:
    from ..params.base import StructuredParams


@dataclass
class OptimizationRequest:
    """Optimization request container.

    This class wraps user input and provides a session identifier
    for tracking long-running optimizations.

    Attributes:
        user_input: User-provided parameters (from frontend JSON)
        structured_params: Pre-computed structured parameters (optional)
            If provided, bypasses the wizard and uses these directly.
        session_id: Unique identifier for this optimization session
        metadata: Additional metadata (e.g., client info, timestamp)

    Example:
        # From frontend JSON
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

        # With pre-computed params (advanced usage)
        request = OptimizationRequest(
            user_input=None,
            structured_params=my_fft_params,
            session_id='custom-session-123'
        )
    """
    user_input: Optional[Dict[str, Any]] = None
    structured_params: Optional['StructuredParams'] = None
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate request has either user_input or structured_params."""
        if self.user_input is None and self.structured_params is None:
            raise ValueError("Request must have either user_input or structured_params")

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'OptimizationRequest':
        """Create request from JSON data.

        Args:
            data: JSON dictionary with user input

        Returns:
            OptimizationRequest instance
        """
        session_id = data.pop('session_id', None) or str(uuid.uuid4())
        metadata = data.pop('metadata', {})

        return cls(
            user_input=data,
            structured_params=None,
            session_id=session_id,
            metadata=metadata
        )

    @classmethod
    def from_structured_params(
        cls,
        params: 'StructuredParams',
        session_id: Optional[str] = None
    ) -> 'OptimizationRequest':
        """Create request from pre-computed structured parameters.

        This bypasses the wizard layer for advanced users who want
        direct control over optimization parameters.

        Args:
            params: Pre-computed structured parameters
            session_id: Optional session identifier

        Returns:
            OptimizationRequest instance
        """
        return cls(
            user_input=None,
            structured_params=params,
            session_id=session_id or str(uuid.uuid4())
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary for logging/serialization."""
        result = {
            'session_id': self.session_id,
            'metadata': self.metadata,
        }
        if self.user_input:
            result['user_input'] = self.user_input
        if self.structured_params:
            result['structured_params'] = 'StructuredParams(present)'
        return result

    @property
    def has_user_input(self) -> bool:
        """Check if request has user input (needs wizard processing)."""
        return self.user_input is not None

    @property
    def has_structured_params(self) -> bool:
        """Check if request has pre-computed params (skip wizard)."""
        return self.structured_params is not None

    @property
    def doe_type(self) -> Optional[str]:
        """Get DOE type from user input."""
        if self.user_input:
            return self.user_input.get('doe_type')
        return None
