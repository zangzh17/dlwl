"""
Test ASM optimization through the pipeline.
"""
import numpy as np
import torch
import sys
sys.path.insert(0, 'C:/Code/dlwl')

from doe_optimizer.pipeline.runner import run_optimization
from doe_optimizer.pipeline.progress import ProgressInfo


def test_asm_optimization():
    """Test full ASM optimization pipeline."""
    print("=" * 60)
    print("Testing ASM optimization pipeline")
    print("=" * 60)

    # Create a simple target pattern
    target_size = 256
    target = np.zeros((target_size, target_size), dtype=np.float32)

    # Create a 3x3 splitter pattern
    positions = [-1, 0, 1]
    center = target_size // 2
    offset = target_size // 8  # Spacing between spots
    for i in positions:
        for j in positions:
            y = center + i * offset
            x = center + j * offset
            if 0 <= y < target_size and 0 <= x < target_size:
                target[y, x] = 1.0

    # Normalize target
    target = target / target.max() if target.max() > 0 else target

    # Create request data matching what the web frontend sends
    request_data = {
        'target_pattern': target.tolist(),
        'wavelength': 532e-9,
        'device_diameter': 256e-6,
        'pixel_size': 1e-6,
        'propagation_type': 'asm',
        'working_distance': 0.01,  # 10mm
        'target_span': 0.001,  # 1mm
        'doe_pixels': [256, 256],
        'optimization': {
            'phase_method': 'SGD',
            'phase_lr': 3e-9,
            'phase_iters': 100,  # Short test
            'loss_type': 'L2',
        },
        'advanced': {
            'target_margin': 0.1,
            'progress_interval': 10,
        }
    }

    print(f"Target shape: {target.shape}")
    print(f"Target non-zero count: {np.count_nonzero(target)}")

    # Progress callback
    def progress_callback(info: ProgressInfo):
        print(f"  Iter {info.current_iter}/{info.total_iters}, loss={info.current_loss:.6f}")

    print("\nRunning optimization...")
    try:
        response = run_optimization(
            request_data,
            progress_callback=progress_callback,
            max_resolution=2000
        )

        print(f"\nResponse success: {response.success}")
        if response.errors:
            print(f"Errors: {[e.message for e in response.errors]}")
        if response.result:
            print(f"Result keys: {list(response.result.to_dict().keys())}")

    except Exception as e:
        import traceback
        print(f"\nException: {type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    test_asm_optimization()
