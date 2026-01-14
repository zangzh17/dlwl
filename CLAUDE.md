# DOE Optimizer v3.5

Diffractive Optical Element (DOE) optimizer with web interface.

## Quick Start

```bash
# Start web interface
uv run uvicorn web_frontend.backend.app:app --reload
# Open http://localhost:8000
```

## Directory Structure

```
dlwl/
├── doe_optimizer/          # Core library
│   ├── core/               # Propagation, loss, optimizer
│   ├── evaluation/         # Metrics, re-evaluation
│   ├── pipeline/           # OptimizationRunner
│   ├── wizard/             # Parameter generators
│   └── params/             # Structured parameters
├── web_frontend/
│   ├── backend/            # FastAPI (routes/, services/)
│   └── frontend/           # HTML/JS/CSS
└── docs/                   # Documentation
```

## Propagation Types

| Type | Use Case | Periodic |
|------|----------|----------|
| FFT | Far-field, k-space | Yes |
| ASM | Near-field, target ~ DOE size | No |
| SFR | Far-field, adjustable output | No |

## Key Architecture: Wizard/Pipeline Decoupling

```
Wizard → target_pattern → Optimization → Results
```

**Pipeline only sees `target_pattern`**, not wizard-specific concepts (order positions, grid mode, etc.).

## Analysis Upsample (Re-evaluation)

Simulates finer DOE pixels while keeping output unchanged:

```python
# Correct: upsample INPUT phase, keep output grid
phase_upsampled = ndimage.zoom(phase, k, order=0)  # Input larger
eff_pixel_size = pixel_size / k                     # Pixels smaller
output_resolution = (h_orig, w_orig)                # Output unchanged
```

**NOT**: increasing output resolution.

## Common Pitfalls

### 1. SFR target_size must include margin

```python
# Wrong
target_size = (target_span, target_span)

# Correct
margin_factor = 1.0 + target_margin  # default 0.1
target_size = (target_span * margin_factor, target_span * margin_factor)
```

### 2. Analysis Upsample: input vs output

- **Correct**: Upsample input phase, keep output resolution
- **Wrong**: Increase output resolution

### 3. FFT k-space cropping

When upsampling FFT, crop output centered on DC:
```python
center = h_full // 2
start = center - h_orig // 2
cropped = output[start:start + h_orig, ...]
```

## Troubleshooting

### CUDA Error: unknown error

```bash
taskkill /F /IM python.exe   # Kill all Python
nvidia-smi                    # Reset CUDA state
uv run uvicorn ...           # Restart server
```

### Browser Caching

Hard refresh: Ctrl+Shift+R

## Documentation and Others

- `docs/changelog.md` - Version history
- `dlwl-legacy` - old-version project
