# DOE Optimizer

Diffractive Optical Element (DOE) design and fabrication optimization library.

## Overview

This package provides a two-step optimization pipeline for DOE design:

1. **Phase/Height Optimization**: Traditional DOE design to achieve target optical patterns
2. **Fabrication Optimization**: OPE (Optical Proximity Effect) correction for laser direct writing

## Supported DOE Types

| Type | Description |
|------|-------------|
| `SPLITTER_1D` | 1D beam splitter (line array) |
| `SPLITTER_2D` | 2D beam splitter (spot array) |
| `SPOT_PROJECTOR` | Gaussian spot array projector |
| `DIFFUSER` | Uniform intensity diffuser |
| `LENS` | Diffractive focusing lens |
| `LENS_ARRAY` | Microlens array |
| `DEFLECTOR` | Beam deflector (blazed grating) |
| `CUSTOM` | Custom pattern from image |

## Propagation Models

| Model | Use Case |
|-------|----------|
| `ASM` | Near-field (Angular Spectrum Method) |
| `FFT` | Far-field infinite distance (Fraunhofer) |
| `SFR` | Far-field with controllable output size (Scaled Fresnel) |

## Quick Start

```python
from doe_optimizer import (
    DOEType, PhysicalParams, DeviceParams,
    OptimizationParams, TargetParams, DOEConfig, optimize_doe
)

# Configure physical parameters
physical = PhysicalParams(
    wavelength=532e-9,          # 532nm
    refraction_index=1.46,      # Fused silica
    working_distance=None       # None = far-field
)

# Configure device parameters
device = DeviceParams(
    diameter=1e-3,              # 1mm DOE diameter
    pixel_size=1e-6             # 1um pixel size
)

# Configure target pattern
target = TargetParams(
    target_type='angle',        # Angle-based target
    target_span=(0.1, 0.1),     # ±0.1 rad FOV
    tolerance=0.01,             # 1% tolerance
    num_spots=(5, 5)            # 5x5 beam splitter
)

# Create configuration
config = DOEConfig(
    doe_type=DOEType.SPLITTER_2D,
    physical=physical,
    device=device,
    target=target
)

# Run optimization
result = optimize_doe(config)

# Access results
print(f"Metrics: {result.metrics}")
height = result.height          # Height profile [H, W] in meters
phase = result.phase            # Phase profile [H, W] in radians
```

## Configuration Reference

### PhysicalParams

| Parameter | Type | Description |
|-----------|------|-------------|
| `wavelength` | float or list | Working wavelength(s) in meters |
| `refraction_index` | float or list | Material refractive index(es) |
| `working_distance` | float, list, or None | Working distance(s) in meters. None = infinite (far-field) |

### DeviceParams

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `diameter` | float | - | DOE diameter in meters |
| `shape` | str | "square" | Aperture shape: "square" or "circular" |
| `pixel_size` | float | 0.5e-6 | Fabrication pixel size in meters |

### TargetParams

| Parameter | Description |
|-----------|-------------|
| `target_type` | "angle" (far-field) or "size" (near-field) |
| `target_span` | (x, y) span in radians (angle) or meters (size) |
| `tolerance` | Position tolerance as fraction (e.g., 0.05 = 5%) |
| `num_spots` | (rows, cols) for splitter/spot array |
| `diffuser_shape` | "square" or "circular" for diffuser |
| `focal_length` | Focal length in meters for lens |
| `lens_type` | "normal", "cylindrical_x", "cylindrical_y" |
| `array_size` | (rows, cols) for lens array |
| `deflection_angle` | (theta_x, theta_y) in radians for deflector |
| `target_image` | Image array for custom pattern |
| `roi_resolution` | Output region resolution |
| `output_size` | Physical output size for SFR mode |

### OptimizationParams

| Parameter | Default | Description |
|-----------|---------|-------------|
| `phase_method` | "SGD" | Algorithm: "SGD", "GS", "BS" |
| `phase_iters` | 2500 | Number of iterations |
| `phase_lr` | 0.3 | Learning rate |
| `phase_pixel_multiplier` | 1 | Pixel size multiplier for faster optimization |
| `loss_type` | "L2" | Loss function: "L1" or "L2" |
| `optimizer_type` | "adam" | Optimizer: "adam" or "sgd" |

## Tolerance and Pixel Multiplier

The package automatically computes physical limits based on sampling theory:

- **Tolerance limit**: Minimum achievable position tolerance
  - Far-field: `T_limit = λ / (2 * D * Δsin(θ))`
  - Near-field: `T_limit = λ * z / (2 * D * S)`

- **Pixel multiplier options**: Valid downsampling factors
  - `N_max = floor(p_limit / p_global)` where `p_limit = λ / (2 * sin(θ_max))`

Access via:
```python
config.tolerance_limit           # Minimum tolerance
config.max_pixel_multiplier      # Maximum multiplier
config.get_pixel_multiplier_options()  # All valid options
```

## Fabrication Optimization

Enable fabrication optimization with calibration data:

```python
from doe_optimizer import FabricationCalibration
import scipy.io as sio

# Load calibration data from MATLAB file
cal_data = sio.loadmat('calibration.mat')

fab_calibration = FabricationCalibration(
    gt_data=cal_data['GT'],      # Grayscale-to-thickness curve
    lp_data=cal_data['LP'],      # Low-pass (MTF) curve
)

config = DOEConfig(
    ...,
    fabrication=fab_calibration,
    enable_fab_optimization=True
)

result = optimize_doe(config)

# Fabrication-corrected results
if result.fab_simulated_intensity is not None:
    print(f"Fab metrics: {result.fab_metrics}")
```

## Evaluation Metrics

Metrics are computed based on DOE type:

| DOE Type | Metrics |
|----------|---------|
| Splitter/Deflector/Spot | Order efficiency mean/std, uniformity |
| Diffuser/Custom | PSNR, SSIM |
| Lens/Lens Array | Encircled energy within Airy disk |

## Package Structure

```
doe_optimizer/
├── core/
│   ├── config.py       # Configuration dataclasses
│   ├── propagation.py  # ASM, FFT, SFR propagation
│   ├── fabrication.py  # Fabrication model (GT/LP)
│   └── optimizer.py    # SGD, GS, BS optimizers
├── patterns/
│   ├── splitter.py     # Beam splitter patterns
│   ├── lens.py         # Lens/lens array patterns
│   ├── diffuser.py     # Diffuser patterns
│   └── ...             # Other pattern generators
├── pipeline/
│   ├── two_step.py     # Main optimization pipeline
│   └── evaluation.py   # Metrics computation
├── utils/
│   ├── fft_utils.py    # ZoomFFT for SFR
│   ├── image_utils.py  # Image processing
│   └── math_utils.py   # Phase/height conversion
└── examples/
    ├── test_splitter.py
    └── test_lens.py
```

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- NumPy, SciPy, scikit-image
