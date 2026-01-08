"""
Optimization response data structures.

This module defines the response format for DOE optimization results.
All data structures are designed for JSON serialization and Plotly visualization.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, TYPE_CHECKING
import numpy as np

from .errors import ValidationError, ValidationWarning

if TYPE_CHECKING:
    from ..visualization.data import VisualizationData


@dataclass
class MetricsData:
    """Evaluation metrics in JSON-serializable format.

    Attributes:
        total_efficiency: Total diffraction efficiency (0-1)
        uniformity: Uniformity metric (0-1, higher is better)
        mean_efficiency: Mean efficiency per spot/order
        std_efficiency: Standard deviation of efficiency
        order_efficiencies: Per-order efficiency values (for splitters)
        psnr: Peak signal-to-noise ratio (for diffusers/custom)
        ssim: Structural similarity index (for diffusers/custom)
    """
    total_efficiency: Optional[float] = None
    uniformity: Optional[float] = None
    mean_efficiency: Optional[float] = None
    std_efficiency: Optional[float] = None
    order_efficiencies: Optional[List[float]] = None
    psnr: Optional[float] = None
    ssim: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                result[key] = value
        return result

    @classmethod
    def from_evaluation_metrics(cls, metrics: Any) -> 'MetricsData':
        """Convert from internal EvaluationMetrics to API format."""
        return cls(
            total_efficiency=getattr(metrics, 'total_efficiency', None),
            uniformity=getattr(metrics, 'uniformity', None),
            mean_efficiency=getattr(metrics, 'mean_efficiency', None),
            std_efficiency=getattr(metrics, 'std_efficiency', None),
            order_efficiencies=getattr(metrics, 'order_efficiencies', None),
            psnr=getattr(metrics, 'psnr', None),
            ssim=getattr(metrics, 'ssim', None),
        )


@dataclass
class DOEResultData:
    """DOE optimization result in JSON-serializable format.

    All array data is converted to nested lists for JSON serialization.
    Physical units are documented for each field.

    Attributes:
        height: Optimized height profile [H, W] in meters
        phase: Phase profile [H, W] in radians (0 to 2*pi)
        target_intensity: Target intensity pattern [H, W] (normalized)
        simulated_intensity: Simulated intensity [H, W] (normalized)
        metrics: Evaluation metrics
        period_pixels: Period size in pixels (for periodic DOEs)
        device_height: Full device height [H, W] in meters
        device_phase: Full device phase [H, W] in radians
        device_phase_with_fresnel: Phase with Fresnel overlay [H, W]
        fresnel_phase: Fresnel phase overlay [H, W] (for finite distance)
        splitter_info: Splitter-specific info (order angles, positions)
        computed_params: Wizard-computed parameters (tolerance_limit, etc.)
    """
    # Core results
    height: List[List[float]]
    phase: List[List[float]]
    target_intensity: List[List[float]]
    simulated_intensity: List[List[float]]
    metrics: MetricsData

    # Optional full device representation
    period_pixels: Optional[int] = None
    device_height: Optional[List[List[float]]] = None
    device_phase: Optional[List[List[float]]] = None
    device_phase_with_fresnel: Optional[List[List[float]]] = None
    fresnel_phase: Optional[List[List[float]]] = None

    # Splitter-specific info
    splitter_info: Optional[Dict[str, Any]] = None

    # Computed parameters from wizard
    computed_params: Optional[Dict[str, Any]] = None

    # Fabrication optimization results
    fab_simulated_intensity: Optional[List[List[float]]] = None
    fab_metrics: Optional[MetricsData] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            'height': self.height,
            'phase': self.phase,
            'target_intensity': self.target_intensity,
            'simulated_intensity': self.simulated_intensity,
            'metrics': self.metrics.to_dict(),
        }

        # Add optional fields if present
        optional_fields = [
            'period_pixels', 'device_height', 'device_phase',
            'device_phase_with_fresnel', 'fresnel_phase',
            'splitter_info', 'computed_params',
            'fab_simulated_intensity'
        ]
        for field_name in optional_fields:
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value

        if self.fab_metrics:
            result['fab_metrics'] = self.fab_metrics.to_dict()

        return result

    @classmethod
    def from_arrays(
        cls,
        height: np.ndarray,
        phase: np.ndarray,
        target_intensity: np.ndarray,
        simulated_intensity: np.ndarray,
        metrics: 'MetricsData',
        **kwargs
    ) -> 'DOEResultData':
        """Create from numpy arrays (converts to lists).

        Args:
            height: Height profile array
            phase: Phase profile array
            target_intensity: Target intensity array
            simulated_intensity: Simulated intensity array
            metrics: Evaluation metrics
            **kwargs: Additional optional fields

        Returns:
            DOEResultData instance with list data
        """
        # Convert numpy arrays to lists
        def to_list(arr):
            if isinstance(arr, np.ndarray):
                return arr.tolist()
            return arr

        return cls(
            height=to_list(height),
            phase=to_list(phase),
            target_intensity=to_list(target_intensity),
            simulated_intensity=to_list(simulated_intensity),
            metrics=metrics,
            period_pixels=kwargs.get('period_pixels'),
            device_height=to_list(kwargs.get('device_height')),
            device_phase=to_list(kwargs.get('device_phase')),
            device_phase_with_fresnel=to_list(kwargs.get('device_phase_with_fresnel')),
            fresnel_phase=to_list(kwargs.get('fresnel_phase')),
            splitter_info=kwargs.get('splitter_info'),
            computed_params=kwargs.get('computed_params'),
            fab_simulated_intensity=to_list(kwargs.get('fab_simulated_intensity')),
            fab_metrics=kwargs.get('fab_metrics'),
        )


@dataclass
class OptimizationResponse:
    """Complete optimization response.

    This is the main response structure returned to the frontend.

    Attributes:
        success: Whether optimization completed successfully
        session_id: Session identifier from the request
        result: Optimization result data (if successful)
        visualization: Visualization data for Plotly (if successful)
        errors: Validation/runtime errors (if failed)
        warnings: Non-fatal warnings
        progress: Final progress information
    """
    success: bool
    session_id: str
    result: Optional[DOEResultData] = None
    visualization: Optional['VisualizationData'] = None
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationWarning] = field(default_factory=list)
    progress: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        response = {
            'success': self.success,
            'session_id': self.session_id,
        }

        if self.result:
            response['result'] = self.result.to_dict()

        if self.visualization:
            response['visualization'] = self.visualization.to_plotly_json()

        if self.errors:
            response['errors'] = [e.to_dict() for e in self.errors]

        if self.warnings:
            response['warnings'] = [w.to_dict() for w in self.warnings]

        if self.progress:
            response['progress'] = self.progress

        return response

    @classmethod
    def success_response(
        cls,
        session_id: str,
        result: DOEResultData,
        visualization: Optional['VisualizationData'] = None,
        warnings: Optional[List[ValidationWarning]] = None
    ) -> 'OptimizationResponse':
        """Create a successful response.

        Args:
            session_id: Session identifier
            result: Optimization result
            visualization: Optional visualization data
            warnings: Optional warnings

        Returns:
            OptimizationResponse with success=True
        """
        return cls(
            success=True,
            session_id=session_id,
            result=result,
            visualization=visualization,
            warnings=warnings or [],
        )

    @classmethod
    def error_response(
        cls,
        session_id: str,
        errors: List[ValidationError],
        warnings: Optional[List[ValidationWarning]] = None
    ) -> 'OptimizationResponse':
        """Create an error response.

        Args:
            session_id: Session identifier
            errors: List of errors
            warnings: Optional warnings

        Returns:
            OptimizationResponse with success=False
        """
        return cls(
            success=False,
            session_id=session_id,
            errors=errors,
            warnings=warnings or [],
        )
