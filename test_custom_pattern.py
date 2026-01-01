"""Test custom pattern optimization (based on gauss.yml config)."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from doe_optimizer import (
    DOEType,
    PhysicalParams,
    DeviceParams,
    OptimizationParams,
    TargetParams,
    DOEConfig,
    optimize_doe,
)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load or create target pattern
    data_dir = Path(__file__).parent / 'data'
    image_path = data_dir / 'gauss.jpg'

    if image_path.exists():
        print(f"Loading image: {image_path}")
        from PIL import Image
        img = Image.open(image_path).convert('L')
        target_image = np.array(img, dtype=np.float32) / 255.0
    else:
        print("Creating Gaussian test pattern...")
        size = 128
        x = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, x)
        target_image = np.exp(-(X**2 + Y**2) / 0.3)

    # Physical parameters (ref: gauss.yml)
    physical = PhysicalParams(
        wavelength=650e-9,       # 650nm red
        refraction_index=1.62,   # Photoresist
        working_distance=0.5e-3,  # 1mm propagation distance
    )

    # Device parameters
    device_params = DeviceParams(
        diameter=256e-6,         # 256um DOE diameter
        pixel_size=1e-6,
    )

    # Target parameters
    target = TargetParams(
        target_type='size',
        target_span=(100e-6, 100e-6),  # 100um x 100um output area
        tolerance=0.01,
        target_image=target_image,
        roi_resolution=(256, 256),
    )

    # Optimization parameters (ref: gauss.yml)
    # Note: lr=1e-8 is for height in meters (um scale)
    optimization = OptimizationParams(
        phase_method='SGD',
        phase_iters=1000,
        phase_lr=1e-8,
        loss_type='L2',
    )

    # Create config
    config = DOEConfig(
        doe_type=DOEType.CUSTOM,
        physical=physical,
        device=device_params,
        target=target,
        optimization=optimization,
    )

    print(f"\nDOE resolution: {config.slm_resolution}")
    print(f"Propagation model: {config.prop_model.value}")
    print(f"Tolerance limit: {config.tolerance_limit:.4f}")

    # Run optimization
    print("\nStarting optimization...")
    result = optimize_doe(config, device=device)

    print(f"\nFinal results:")
    print(f"  Height range: [{result.height.min()*1e6:.3f}, {result.height.max()*1e6:.3f}] um")
    print(f"  Phase range: [{result.phase.min():.3f}, {result.phase.max():.3f}] rad")
    if result.metrics.psnr is not None:
        print(f"  PSNR: {result.metrics.psnr:.2f} dB")
        print(f"  SSIM: {result.metrics.ssim:.4f}")

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Phase distribution
    ax = axes[0, 0]
    im = ax.imshow(result.phase, cmap='twilight')
    ax.set_title('Phase Distribution (rad)')
    plt.colorbar(im, ax=ax)

    # Height distribution
    ax = axes[0, 1]
    im = ax.imshow(result.height * 1e6, cmap='viridis')
    ax.set_title('Height Distribution (um)')
    plt.colorbar(im, ax=ax)

    # Target pattern
    ax = axes[1, 0]
    im = ax.imshow(result.target_intensity, cmap='gray')
    ax.set_title('Target Pattern')
    plt.colorbar(im, ax=ax)

    # Simulated pattern
    ax = axes[1, 1]
    im = ax.imshow(result.simulated_intensity, cmap='gray')
    psnr_str = f'{result.metrics.psnr:.1f}' if result.metrics.psnr else 'N/A'
    ax.set_title(f'Simulated Pattern (PSNR: {psnr_str}dB)')
    plt.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig('custom_pattern_result.png', dpi=150)
    print("\nResult saved to custom_pattern_result.png")
    plt.show()


if __name__ == '__main__':
    main()
