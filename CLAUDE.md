# DOE Optimizer v3.3

A Python backend for designing and optimizing Diffractive Optical Elements (DOEs) with a web-based testing interface.

## Project Overview

**Goal**: Optimize DOE phase distributions to produce target optical patterns via light propagation simulation.

**Two-Step Method**:
1. **Phase Optimization**: Optimize DOE height/phase using propagation models (FFT/ASM/SFR)
2. **Fabrication Optimization** (optional): Optimize laser writing parameters to match target morphology

## Directory Structure

```
dlwl/
├── doe_optimizer/          # Core library
│   ├── api/                # JSON request/response schemas
│   ├── params/             # Structured parameters (FFT/SFR/ASM)
│   ├── wizard/             # Parameter generators (Splitter/Diffuser/Lens)
│   ├── validation/         # Parameter validation
│   ├── core/               # Propagation, loss functions, optimizer
│   ├── pipeline/           # OptimizationRunner, progress reporting
│   ├── evaluation/         # Result evaluation, metrics, re-evaluation
│   │   └── reevaluate.py   # Unified re-evaluation at different resolutions
│   ├── visualization/      # Plotly data export
│   └── utils/              # FFT, image, math utilities
├── web_frontend/           # Web interface (v3.0)
│   ├── backend/            # FastAPI backend
│   │   ├── app.py          # Main application
│   │   ├── routes/         # API endpoints
│   │   └── services/       # Task manager, preview service
│   └── frontend/           # HTML/CSS/JS frontend
│       ├── index.html      # Single-page application
│       ├── css/            # Styles
│       └── js/             # JavaScript modules
├── test_splitter_v2.py     # Test scripts
├── test_splitter_validation.py   # Test scripts
├── results/                # Test results
├── docs/                   # Documentation
├── data/                   # Input images, calibration data
└── archive_v1/             # Legacy code (reference only)
```

## Web Frontend (v3.0)

Start the web interface:
```bash
uv run uvicorn web_frontend.backend.app:app --reload
```

Open http://localhost:8000 in your browser to access:
- **Wizard**: Configure DOE parameters interactively
- **Preview**: Visualize geometry and target patterns
- **Optimize**: Run optimization with real-time progress
- **Results**: View results with Plotly charts and export

### API Endpoints
- `POST /api/wizard` - Generate structured params from wizard input
- `POST /api/validate` - Validate parameters
- `POST /api/preview` - Generate preview (geometry SVG + target plots)
- `POST /api/optimize` - Start optimization (returns task_id)
- `GET /api/status/{id}` - Query progress
- `POST /api/cancel/{id}` - Cancel optimization
- `GET /api/result/{id}` - Get results
- `WebSocket /api/ws/optimize/{id}` - Real-time progress streaming

### Reference Values (Read-only)
The frontend displays computed reference values to help users understand the optical limits:
- **Period**: Physical size of one period (um)
- **DOE Pixel**: Physical fabrication pixel size (um)
- **Max Angle**: Maximum diffraction angle = arcsin(λ / 2×pixel_effective), where pixel_effective = pixel_size × pixel_multiplier
- **Diff. Limit**: Angular resolution = λ/D (deg)
- **Min Tolerance**: Minimum tolerance when period = device diameter
- **Max Mult.**: Maximum pixel_multiplier before pattern exceeds diffraction limit

## Quick Start (Python API)

```python
from doe_optimizer import run_optimization

# Define DOE parameters
user_input = {
    'doe_type': 'splitter_2d',
    'wavelength': 532e-9,
    'device_diameter': 256e-6,
    'pixel_size': 1e-6,
    'target_spec': {
        'num_spots': [5, 5],
        'target_type': 'angle',
        'target_span': [0.1, 0.1],
        'grid_mode': 'natural'
    }
}

# Run optimization
response = run_optimization(user_input)

if response.success:
    result = response.result
    print(f"Efficiency: {result.metrics['total_efficiency']:.4f}")
```

## Propagation Types

| Type | Use Case | Output | Periodic |
|------|----------|--------|----------|
| **FFT** | Infinite distance / k-space | Angular spectrum | Yes |
| **ASM** | Near-field, target ~ DOE size | Same physical size | No |
| **SFR** | Far-field, large target area | Adjustable output size | No |
| **Periodic+Fresnel** | Strategy 2: finite distance with periodic DOE | Physical coordinates | Yes |

## DOE Types

- **Splitter (1D/2D)**: Beam splitting into spot arrays
- **Diffuser**: Uniform illumination patterns
- **Lens/Lens Array**: Focusing elements
- **Custom**: User-defined target patterns

## Architecture Principles

### CRITICAL: Decoupling Wizards from Core Pipeline

**Wizards are ONLY for generating DOE Settings. The core pipeline (optimization, evaluation, metrics) must NEVER depend on wizard-specific concepts.**

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌─────────┐
│   Wizard    │ ──► │ DOE Settings │ ──► │ Optimization│ ──► │ Results │
│ (splitter,  │     │ (universal)  │     │   (generic) │     │(metrics)│
│  diffuser,  │     │              │     │             │     │         │
│  lens, etc) │     │              │     │             │     │         │
└─────────────┘     └──────────────┘     └─────────────┘     └─────────┘
      │                    │                    │                  │
   Generates          Core data           Only sees           Derived from
   parameters         structure        DOE Settings         target_pattern
```

**DOE Settings (universal):**
```python
{
    'target_pattern': [...],      # 2D target intensity distribution
    'wavelength': 532e-9,
    'pixel_size': 1e-6,
    'propagation_type': 'fft',    # 'fft', 'asm', 'sfr'
    'simulation_pixels': [64, 64],
    # ... other physical params
}
```

**What the pipeline does NOT know:**
- What wizard generated the parameters (splitter? diffuser? custom?)
- "Order positions", "working orders" (splitter-specific concepts)
- Grid mode (natural/uniform) - wizard concepts

**Efficiency Calculation:**
- Target indices are derived from `target_pattern` (non-zero positions)
- No wizard metadata is passed to optimization/evaluation
- `_extract_target_indices(target_pattern)` finds target spots automatically

**Display Metadata (frontend only):**
- Wizard metadata (angles, order indices) is stored in `AppState.metadata`
- Used ONLY for visualization (e.g., showing angles on charts)
- NEVER used for efficiency calculation

**Why this matters:**
1. Adding new DOE types doesn't require changing the pipeline
2. Efficiency calculation works for ANY target pattern
3. Clear separation of concerns
4. Easier testing and maintenance

## Key APIs

### OptimizationRunner
```python
from doe_optimizer import OptimizationRunner, OptimizationRequest

runner = OptimizationRunner(max_resolution=2000)
request = OptimizationRequest.from_json(user_input)
response = runner.run(request, progress_callback=on_progress)
```

### Progress Callback
```python
from doe_optimizer import ProgressInfo, CancellationToken

cancel_token = CancellationToken()

def on_progress(info: ProgressInfo):
    print(f"[{info.stage}] {info.progress:.1%}, Loss: {info.current_loss:.2e}")
    if should_cancel:
        cancel_token.cancel()
```

### Loss Functions
```python
from doe_optimizer import create_loss

loss_fn = create_loss('L2')  # or 'L1', 'focal_efficiency'
```

## Splitter-Specific Concepts

### Grid Modes
- **Natural**: k-space uniform, follows diffraction orders
- **Uniform**: Angle-space uniform, requires tolerance parameter

### Finite Distance Strategies
- **Strategy 1 (ASM/SFR)**: Direct propagation when target ~ DOE size
- **Strategy 2 (Periodic+Fresnel)**: Periodic DOE + Fresnel lens overlay

### Key Parameters
- `tolerance`: Angular error tolerance (%) for uniform grid
- `pixel_size`: Fabrication pixel size (determines max diffraction angle: θ_max = arcsin(λ/2p))
- `pixel_multiplier`: Groups DOE pixels, effectively increasing pixel size (reduces max angle)
- `period_pixels`: Optimization unit size - one period of the periodic pattern
- `doe_pixels`: Full device size in pixels
- `num_periods`: Derived value = doe_pixels / period_pixels (how many periods tile the device)
- `simulation_upsample`: Resolution multiplier during optimization
- `analysis_upsample`: Resolution multiplier for post-optimization evaluation

## Testing

Run comprehensive tests:
```
uv run test_splitter_v2.py
uv run test_splitter_validation.py
```

Test cases cover:
- FFT 1D/2D with natural/uniform grids
- ASM/SFR finite distance evaluation
- Strategy 2 (Periodic+Fresnel) splitters
- Upsampling during optimization and evaluation
- parameter validation

## Key Implementation Notes

### Efficiency Calculation
- Target indices derived from `target_pattern` (non-zero positions)
- FFT k-space: Single pixel sampling at target indices
- Physical (ASM/SFR): Airy disk integration around target indices

### Analysis Upsample (Re-evaluation)

Unified re-evaluation module: `doe_optimizer/evaluation/reevaluate.py`

**Core concept**: Upsampling k× means k² more samples covering the SAME physical/angular range.
- Device size unchanged
- Pixel count increases
- Effective pixel size decreases: `effective_pixel_size = original_pixel_size / upsample_factor`

**FFT propagation**:
- Uses **tiling**: `np.tile(phase, (N, N))` - more periods = sharper diffraction
- Target indices scale relative to center: `new_pos = center_new + (old_pos - center_old) * N`
- Same angular range, sharper k-space peaks

**ASM/SFR propagation**:
- Uses **interpolation**: `scipy.ndimage.zoom(phase, N, order=3)`
- Propagate with smaller feature_size (effective_pixel_size)
- Same physical range, more output samples

**Usage**:
```python
from doe_optimizer.evaluation import reevaluate_at_resolution

result = reevaluate_at_resolution(
    phase=phase_np,
    target=target_np,
    upsample_factor=2,
    propagation_type='fft',  # 'fft', 'asm', 'sfr', 'periodic_fresnel'
    pixel_size=1e-6,
    wavelength=532e-9
)
# result.phase - tiled/upsampled phase
# result.effective_pixel_size - pixel_size / upsample_factor
```

### Upsampling Normalization
- 1D arrays: Normalize by `upsample_factor^2`
- 2D arrays: Normalize by `upsample_factor^4`

### Energy Conservation
- FFT output energy scales as `(N*M)^2` for N×M input
- Cropped output requires energy renormalization for fair comparison

## Documentation

See `docs/` for detailed information:
- `SFR_theory.md`: Scalable Fourier Representation theory
- `python_service.md`: Service API documentation
- `changelog.md`: Version history

## Dependencies

- Use uv for environment management: `uv sync`
- Core: torch, numpy, scipy, plotly
- Web: fastapi, uvicorn, websockets

## Troubleshooting

### CUDA Error: unknown error (recurring issue)

This error occurs when CUDA gets into a bad state. **Not a code bug** - it's a CUDA driver/state issue.

**Solution:**
```bash
# 1. Kill all Python processes
taskkill /F /IM python.exe

# 2. Check GPU status (helps reset CUDA)
nvidia-smi

# 3. Restart server (must use GPU, not CPU-only)
uv run uvicorn web_frontend.backend.app:app --reload
```

**DO NOT** use `CUDA_VISIBLE_DEVICES=` to disable GPU - optimization requires GPU.

The codebase has built-in CUDA error recovery:
- `wizard/base.py`: `_get_device()` tests CUDA before use
- `routes/wizard.py`: `_generate_params_with_cuda_recovery()` falls back gracefully

### Browser Caching Issues

When frontend JS/CSS changes don't take effect:
1. Hard refresh: Ctrl+Shift+R (or Cmd+Shift+R on Mac)
2. The backend auto-versions static files using mtime (see `backend/app.py`)
3. Check DevTools Network tab - ensure files are being fetched, not cached

---

*v3.3 - Unified re-evaluation module for all propagation types (FFT tiling, ASM/SFR interpolation)*
*v3.2 - Decoupled wizards from core pipeline: efficiency calculation now derives target_indices from target_pattern*
*v3.1 - Added Strategy 2 (Periodic+Fresnel) propagation, improved UI with Reference Values*
*v3.0 - Added web frontend for interactive testing (FastAPI + Plotly)*
*v2.0 - Refactored architecture with layered design (API -> Wizard -> Validation -> Core -> Evaluation)*
