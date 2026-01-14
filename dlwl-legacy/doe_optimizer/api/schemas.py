"""
JSON schema definitions for frontend communication.

This module defines TypedDict schemas for all user input parameters.
These schemas are designed for JSON serialization/deserialization.
"""

from typing import TypedDict, Optional, List, Union, Literal
from enum import Enum


# =============================================================================
# Enum Types (as string literals for JSON compatibility)
# =============================================================================

DOETypeSchema = Literal[
    "splitter_1d",
    "splitter_2d",
    "spot_projector",
    "diffuser",
    "lens",
    "lens_array",
    "deflector",
    "custom"
]

SplitterModeSchema = Literal["natural", "uniform"]

TargetTypeSchema = Literal["angle", "size"]

DeviceShapeSchema = Literal["square", "circular"]

LossTypeSchema = Literal["L1", "L2", "focal_efficiency", "focal_uniformity"]

OptMethodSchema = Literal["SGD", "GS", "BS"]


# =============================================================================
# Target Specification Schemas (DOE-type specific)
# =============================================================================

class SplitterTargetSpec(TypedDict, total=False):
    """Target specification for splitter/spot projector DOEs.

    Attributes:
        num_spots: Number of spots (int for 1D, tuple for 2D)
        target_type: 'angle' for angular specification, 'size' for physical size
        target_span: Span in radians (angle) or meters (size)
        grid_mode: 'natural' for diffraction-order aligned, 'uniform' for equal spacing
        tolerance: Acceptable angle error (0-1), required for 'uniform' mode
    """
    num_spots: Union[int, List[int]]  # [ny, nx] for 2D
    target_type: TargetTypeSchema
    target_span: Union[float, List[float]]  # [span_y, span_x] for 2D
    grid_mode: SplitterModeSchema
    tolerance: float  # Required for uniform mode


class DiffuserTargetSpec(TypedDict, total=False):
    """Target specification for diffuser DOEs.

    Attributes:
        shape: 'square' or 'circular' diffusion pattern
        target_type: 'angle' or 'size'
        target_span: Full angle or size of the diffused beam
    """
    shape: Literal["square", "circular"]
    target_type: TargetTypeSchema
    target_span: Union[float, List[float]]


class LensTargetSpec(TypedDict, total=False):
    """Target specification for lens DOEs.

    Attributes:
        focal_length: Focal length in meters
        lens_type: 'normal' for 2D lens, 'cylindrical_x/y' for 1D
        array_size: For lens arrays, number of lenses [ny, nx]
    """
    focal_length: float
    lens_type: Literal["normal", "cylindrical_x", "cylindrical_y"]
    array_size: Optional[List[int]]  # For lens array


class DeflectorTargetSpec(TypedDict, total=False):
    """Target specification for deflector DOEs.

    Attributes:
        deflection_angle: Deflection angle in radians [theta_y, theta_x]
    """
    deflection_angle: List[float]


class CustomTargetSpec(TypedDict, total=False):
    """Target specification for custom pattern DOEs.

    Attributes:
        image_data: Base64 encoded image or file path
        target_resolution: Output resolution [H, W]
        target_type: 'angle' or 'size'
        target_span: Span of the target pattern
        tolerance: For period estimation
    """
    image_data: str  # Base64 or path
    target_resolution: List[int]
    target_type: TargetTypeSchema
    target_span: Union[float, List[float]]
    tolerance: float


# Union type for all target specs
TargetSpecSchema = Union[
    SplitterTargetSpec,
    DiffuserTargetSpec,
    LensTargetSpec,
    DeflectorTargetSpec,
    CustomTargetSpec,
]


# =============================================================================
# Optimization Parameters Schema
# =============================================================================

class OptimizationParamsSchema(TypedDict, total=False):
    """Optimization algorithm parameters.

    Attributes:
        phase_method: Algorithm for phase optimization
        phase_lr: Learning rate for phase optimization
        phase_iters: Number of iterations
        phase_pixel_multiplier: Pixel size multiplier for optimization
        fab_enabled: Enable fabrication optimization
        fab_method: Algorithm for fab optimization
        fab_lr: Learning rate for fab optimization
        fab_iters: Number of fab iterations
        loss_type: Loss function type
        simulation_upsample: Upsampling factor for simulation
    """
    phase_method: OptMethodSchema
    phase_lr: float
    phase_iters: int
    phase_pixel_multiplier: int
    fab_enabled: bool
    fab_method: OptMethodSchema
    fab_lr: float
    fab_iters: int
    loss_type: LossTypeSchema
    simulation_upsample: int


# =============================================================================
# Main User Input Schema
# =============================================================================

class UserInputSchema(TypedDict, total=False):
    """Complete user input schema for DOE optimization.

    This is the main interface for frontend communication.
    All parameters use SI units (meters, radians) unless otherwise specified.

    Attributes:
        doe_type: Type of DOE to design
        wavelength: Working wavelength in meters
        working_distance: Propagation distance (None for infinite)
        device_diameter: DOE diameter in meters
        device_shape: Aperture shape
        pixel_size: Fabrication pixel size in meters
        refraction_index: Material refractive index
        target_spec: DOE-type-specific target parameters
        optimization: Optimization algorithm parameters
        fab_recipe: Fabrication recipe name (for OPE correction)
    """
    # Required fields
    doe_type: DOETypeSchema
    wavelength: float
    device_diameter: float
    pixel_size: float
    target_spec: TargetSpecSchema

    # Optional fields with defaults
    working_distance: Optional[float]  # None = infinite
    device_shape: DeviceShapeSchema  # Default: 'square'
    refraction_index: float  # Default: 1.62
    optimization: OptimizationParamsSchema
    fab_recipe: Optional[str]


# =============================================================================
# Helper Functions
# =============================================================================

def validate_user_input(data: dict) -> List[str]:
    """Basic validation of user input structure.

    Args:
        data: Input dictionary to validate

    Returns:
        List of error messages (empty if valid)
    """
    errors = []

    # Required fields
    required = ['doe_type', 'wavelength', 'device_diameter', 'pixel_size', 'target_spec']
    for field in required:
        if field not in data:
            errors.append(f"Missing required field: {field}")

    if 'doe_type' in data:
        valid_types = ['splitter_1d', 'splitter_2d', 'spot_projector', 'diffuser',
                       'lens', 'lens_array', 'deflector', 'custom']
        if data['doe_type'] not in valid_types:
            errors.append(f"Invalid doe_type: {data['doe_type']}")

    # Numeric constraints
    if 'wavelength' in data and data['wavelength'] <= 0:
        errors.append("wavelength must be positive")

    if 'device_diameter' in data and data['device_diameter'] <= 0:
        errors.append("device_diameter must be positive")

    if 'pixel_size' in data and data['pixel_size'] <= 0:
        errors.append("pixel_size must be positive")

    return errors


def get_default_optimization_params(doe_type: str) -> OptimizationParamsSchema:
    """Get default optimization parameters for a DOE type.

    Args:
        doe_type: Type of DOE

    Returns:
        Default optimization parameters
    """
    defaults: OptimizationParamsSchema = {
        'phase_method': 'SGD',
        'phase_lr': 1e-8,
        'phase_iters': 10000,
        'phase_pixel_multiplier': 1,
        'fab_enabled': False,
        'fab_method': 'SGD',
        'fab_lr': 200.0,
        'fab_iters': 25000,
        'loss_type': 'L2',
        'simulation_upsample': 1,
    }

    # DOE-type specific adjustments
    if doe_type in ['lens', 'lens_array']:
        defaults['loss_type'] = 'focal_efficiency'

    return defaults
