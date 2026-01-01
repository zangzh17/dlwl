"""Test 2D beam splitter optimization with multiple cases.

Demonstrates:
1. Infinite distance, angle-based target (odd spots) - Natural and Uniform grid
2. Infinite distance, angle-based target (even spots) - Symmetric order selection
3. Finite distance, size-based target (Strategy 2: Periodic + Fresnel)
4. Finite distance with small target (Strategy 1: ASM - if applicable)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

from doe_optimizer import (
    DOEType,
    SplitterMode,
    FiniteDistanceStrategy,
    PhysicalParams,
    DeviceParams,
    OptimizationParams,
    TargetParams,
    DOEConfig,
    optimize_doe,
    MAX_OPTIMIZATION_RESOLUTION,
)
from doe_optimizer.pipeline import (
    evaluate_finite_distance_splitter,
    FiniteDistanceEvaluation,
)
from doe_optimizer.utils import (
    plot_splitter_result,
    plot_comprehensive_splitter_result,
    plot_angular_distribution,
    plot_finite_distance_evaluation,
)


def test_infinite_distance_odd_spots():
    """Test Case 1: Infinite distance, odd spot count (5x5)."""
    print("\n" + "="*70)
    print("CASE 1: Infinite distance, 5x5 spots (odd), Natural grid")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    physical = PhysicalParams(
        wavelength=532e-9,
        refraction_index=1.46,
        working_distance=None,  # Infinite
    )

    device_params = DeviceParams(
        diameter=256e-6,
        pixel_size=1e-6,
    )

    target = TargetParams(
        target_type='angle',
        target_span=(0.2, 0.2),  # +/-0.1 rad FOV
        tolerance=0.01,
        num_spots=(5, 5),  # Odd count
        splitter_mode='natural',
    )

    optimization = OptimizationParams(
        phase_method='SGD',
        phase_iters=1000,
        phase_lr=1e-8,
        loss_type='L2',
    )

    config = DOEConfig(
        doe_type=DOEType.SPLITTER_2D,
        physical=physical,
        device=device_params,
        target=target,
        optimization=optimization,
    )

    params = config.get_splitter_params()
    print(f"\nConfiguration:")
    print(f"  Mode: {params['mode'].value}")
    print(f"  Period: {params['period']*1e6:.2f} um")
    print(f"  Working orders: {len(params['working_orders'])}")
    print(f"  Is periodic: {params['is_periodic']}")
    print(f"  Is even spots: {params['is_even_spots']}")

    result = optimize_doe(config, device=device, verbose=True)

    print(f"\nResults:")
    print(f"  Total Efficiency: {result.metrics.total_efficiency:.4f}")
    print(f"  Uniformity: {result.metrics.order_uniformity:.4f}")
    print(f"  Order mean: {result.metrics.order_efficiency_mean:.4f}")

    return result


def test_infinite_distance_even_spots():
    """Test Case 2: Infinite distance, even spot count (4x4) - symmetric orders."""
    print("\n" + "="*70)
    print("CASE 2: Infinite distance, 4x4 spots (even), Natural grid")
    print("        Testing symmetric order selection with doubled period")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    physical = PhysicalParams(
        wavelength=532e-9,
        refraction_index=1.46,
        working_distance=None,
    )

    device_params = DeviceParams(
        diameter=256e-6,
        pixel_size=1e-6,
    )

    target = TargetParams(
        target_type='angle',
        target_span=(0.2, 0.2),
        tolerance=0.01,
        num_spots=(4, 4),  # Even count - should trigger symmetric selection
        splitter_mode='natural',
    )

    optimization = OptimizationParams(
        phase_method='SGD',
        phase_iters=1000,
        phase_lr=1e-8,
        loss_type='L2',
    )

    config = DOEConfig(
        doe_type=DOEType.SPLITTER_2D,
        physical=physical,
        device=device_params,
        target=target,
        optimization=optimization,
    )

    params = config.get_splitter_params()
    print(f"\nConfiguration:")
    print(f"  Mode: {params['mode'].value}")
    print(f"  Period: {params['period']*1e6:.2f} um (doubled for even spots)")
    print(f"  Working orders: {len(params['working_orders'])}")
    print(f"  Is even spots: {params['is_even_spots']}")
    print(f"  Orders (should be symmetric): {params['working_orders'][:4]}...")

    # Check symmetry
    orders = params['working_orders']
    is_symmetric = all((-o[0], -o[1]) in orders for o in orders)
    print(f"  Pattern is symmetric: {is_symmetric}")

    result = optimize_doe(config, device=device, verbose=True)

    print(f"\nResults:")
    print(f"  Uniformity: {result.metrics.order_uniformity:.4f}")

    return result


def test_finite_distance_large_target():
    """Test Case 3: Finite distance with large target (Strategy 2: Periodic + Fresnel)."""
    print("\n" + "="*70)
    print("CASE 3: Finite distance (100mm), 20mm x 20mm target")
    print("        Testing Strategy 2: Periodic + Fresnel overlay")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    physical = PhysicalParams(
        wavelength=532e-9,
        refraction_index=1.46,
        working_distance=100e-3,  # 100mm
    )

    device_params = DeviceParams(
        diameter=256e-6,
        pixel_size=1e-6,
    )

    # Large target (20mm) >> DOE size (256um)
    # This should trigger Strategy 2 (periodic + Fresnel)
    target = TargetParams(
        target_type='size',
        target_span=(20e-3, 20e-3),  # 20mm x 20mm
        tolerance=0.006,
        num_spots=(6, 6),
        splitter_mode='natural',
    )

    optimization = OptimizationParams(
        phase_method='SGD',
        phase_iters=1500,
        phase_lr=1e-8,
        loss_type='L2',
    )

    config = DOEConfig(
        doe_type=DOEType.SPLITTER_2D,
        physical=physical,
        device=device_params,
        target=target,
        optimization=optimization,
    )

    strategy = config.get_finite_distance_strategy()
    params = config.get_splitter_params()

    print(f"\nConfiguration:")
    print(f"  Working distance: {physical.working_distance*1000:.1f} mm")
    print(f"  Target size: {target.target_span[0]*1000:.1f} mm x {target.target_span[1]*1000:.1f} mm")
    print(f"  Strategy: {strategy.value if strategy else 'None'}")
    print(f"  Mode: {params['mode'].value}")
    print(f"  Period: {params['period']*1e6:.2f} um")
    print(f"  Is periodic: {params['is_periodic']}")

    # Show target positions
    if params['target_positions']:
        print(f"\n  Target positions (mm):")
        for i, (pos_y, pos_x) in enumerate(params['target_positions'][:4]):
            order = params['working_orders'][i]
            print(f"    Order {order}: ({pos_y*1000:.2f}, {pos_x*1000:.2f}) mm")
        print(f"    ... ({len(params['target_positions'])} total)")

    result = optimize_doe(config, device=device, verbose=True)

    print(f"\nResults:")
    print(f"  Uniformity: {result.metrics.order_uniformity:.4f}")
    print(f"  Has Fresnel phase overlay: {result.fresnel_phase is not None}")
    print(f"  Has device_phase_with_fresnel: {result.device_phase_with_fresnel is not None}")

    return result


def test_finite_distance_small_target():
    """Test Case 3b: Finite distance with small target (Strategy 1: ASM)."""
    print("\n" + "="*70)
    print("CASE 3b: Finite distance (1mm), 200um x 200um target")
    print("         Testing Strategy 1: Direct ASM propagation")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    physical = PhysicalParams(
        wavelength=532e-9,
        refraction_index=1.46,
        working_distance=1e-3,  # 1mm - for ~2.5um Airy radius
    )

    device_params = DeviceParams(
        diameter=256e-6,
        pixel_size=1e-6,
    )

    # Small target (200um) < DOE size (256um)
    # Target must fit within output window for ASM
    # This should trigger Strategy 1 (direct ASM)
    target = TargetParams(
        target_type='size',
        target_span=(200e-6, 200e-6),  # 200um x 200um (fits in 256um DOE)
        tolerance=0.01,
        num_spots=(3, 3),  # 3x3 for small target
        splitter_mode='natural',
    )

    optimization = OptimizationParams(
        phase_method='SGD',
        phase_iters=1000,
        phase_lr=1e-8,
        loss_type='L2',
    )

    config = DOEConfig(
        doe_type=DOEType.SPLITTER_2D,
        physical=physical,
        device=device_params,
        target=target,
        optimization=optimization,
    )

    strategy = config.get_finite_distance_strategy()
    params = config.get_splitter_params()

    print(f"\nConfiguration:")
    print(f"  Working distance: {physical.working_distance*1000:.1f} mm")
    print(f"  Target size: {target.target_span[0]*1e6:.1f} um x {target.target_span[1]*1e6:.1f} um")
    print(f"  Strategy: {strategy.value if strategy else 'None'}")
    print(f"  Mode: {params['mode'].value}")
    print(f"  Is periodic: {params['is_periodic']}")
    print(f"  Propagation model: ASM (direct)")

    result = optimize_doe(config, device=device, verbose=True)

    print(f"\nResults:")
    print(f"  Total Efficiency: {result.metrics.total_efficiency:.4f}")
    print(f"  Uniformity: {result.metrics.order_uniformity:.4f}")

    return result


def test_uniform_grid_comparison():
    """Test Case 4: Compare Natural vs Uniform grid at finite distance."""
    print("\n" + "="*70)
    print("CASE 4: Comparison - Natural vs Uniform grid (finite distance)")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    results = {}

    for mode in ['natural', 'uniform']:
        print(f"\n--- Testing {mode.upper()} grid ---")

        physical = PhysicalParams(
            wavelength=532e-9,
            refraction_index=1.46,
            working_distance=100e-3,
        )

        device_params = DeviceParams(
            diameter=256e-6,
            pixel_size=1e-6,
        )

        target = TargetParams(
            target_type='size',
            target_span=(20e-3, 20e-3),
            tolerance=0.006,
            num_spots=(5, 5),
            splitter_mode=mode,
        )

        optimization = OptimizationParams(
            phase_method='SGD',
            phase_iters=1000,
            phase_lr=1e-8,
            loss_type='L2',
        )

        config = DOEConfig(
            doe_type=DOEType.SPLITTER_2D,
            physical=physical,
            device=device_params,
            target=target,
            optimization=optimization,
        )

        params = config.get_splitter_params()
        print(f"  Period: {params['period']*1e6:.2f} um")
        print(f"  Grid resolution: {params['num_orders']}")

        result = optimize_doe(config, device=device, verbose=False)
        results[mode] = result

        print(f"  Uniformity: {result.metrics.order_uniformity:.4f}")
        print(f"  Total Efficiency: {result.metrics.total_efficiency:.4f}")

    return results


def test_1d_splitter():
    """Test 1D beam splitter (line of spots)."""
    print("\n" + "="*70)
    print("CASE 5: 1D Splitter, 7 spots, Natural grid")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = DOEConfig(
        doe_type=DOEType.SPLITTER_1D,
        physical=PhysicalParams(
            wavelength=532e-9,
            refraction_index=1.46,
            working_distance=None,  # Infinite
        ),
        device=DeviceParams(
            diameter=256e-6,
            pixel_size=1e-6,
        ),
        target=TargetParams(
            target_type='angle',
            target_span=(0.15,),  # 1D: single span value
            tolerance=0.01,
            num_spots=(7, 1),  # 7 spots in 1D
            splitter_mode='natural',
        ),
        optimization=OptimizationParams(
            phase_iters=800,
        ),
    )

    params = config.get_splitter_params()
    print(f"\nConfiguration:")
    print(f"  Mode: natural")
    print(f"  Period: {params['period']*1e6:.2f} um")
    print(f"  Working orders: {len(params['working_orders'])}")

    result = optimize_doe(config, device=device, verbose=True)

    print(f"\nResults:")
    print(f"  Uniformity: {result.metrics.order_uniformity:.4f}")
    print(f"  Total Efficiency: {result.metrics.total_efficiency:.4f}")

    return result


def test_large_spot_array():
    """Test larger spot array (10x10)."""
    print("\n" + "="*70)
    print("CASE 6: Large 10x10 spot array, Natural grid")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = DOEConfig(
        doe_type=DOEType.SPLITTER_2D,
        physical=PhysicalParams(
            wavelength=532e-9,
            refraction_index=1.46,
            working_distance=None,  # Infinite
        ),
        device=DeviceParams(
            diameter=512e-6,
            pixel_size=1e-6,
        ),
        target=TargetParams(
            target_type='angle',
            target_span=(0.25, 0.25),
            tolerance=0.005,
            num_spots=(10, 10),  # 100 spots total
            splitter_mode='natural',
        ),
        optimization=OptimizationParams(
            phase_iters=1500,
        ),
    )

    params = config.get_splitter_params()
    print(f"\nConfiguration:")
    print(f"  Period: {params['period']*1e6:.2f} um")
    print(f"  Working orders: {len(params['working_orders'])}")
    print(f"  Is even: {params['is_even_spots']}")

    result = optimize_doe(config, device=device, verbose=True)

    print(f"\nResults:")
    print(f"  Uniformity: {result.metrics.order_uniformity:.4f}")
    print(f"  Total Efficiency: {result.metrics.total_efficiency:.4f}")

    return result


def test_different_wavelengths():
    """Test splitter at different wavelengths."""
    print("\n" + "="*70)
    print("CASE 7: Wavelength comparison (405nm, 532nm, 633nm)")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wavelengths = [405e-9, 532e-9, 633e-9]
    results = {}

    for wl in wavelengths:
        print(f"\n--- Wavelength: {wl*1e9:.0f} nm ---")

        config = DOEConfig(
            doe_type=DOEType.SPLITTER_2D,
            physical=PhysicalParams(
                wavelength=wl,
                refraction_index=1.46,
                working_distance=None,
            ),
            device=DeviceParams(
                diameter=256e-6,
                pixel_size=1e-6,
            ),
            target=TargetParams(
                target_type='angle',
                target_span=(0.15, 0.15),
                tolerance=0.01,
                num_spots=(5, 5),
                splitter_mode='natural',
            ),
            optimization=OptimizationParams(
                phase_iters=800,
            ),
        )

        params = config.get_splitter_params()
        print(f"  Period: {params['period']*1e6:.2f} um")

        result = optimize_doe(config, device=device, verbose=False)
        results[f"{wl*1e9:.0f}nm"] = result

        print(f"  Uniformity: {result.metrics.order_uniformity:.4f}")
        print(f"  Efficiency: {result.metrics.total_efficiency:.4f}")

    return results


def test_1d_finite_distance():
    """Test 1D splitter at finite distance (Strategy 2: Periodic + Fresnel)."""
    print("\n" + "="*70)
    print("CASE 8: 1D Splitter, Finite distance (50mm), 5 spots")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = DOEConfig(
        doe_type=DOEType.SPLITTER_1D,
        physical=PhysicalParams(
            wavelength=532e-9,
            refraction_index=1.46,
            working_distance=50e-3,  # 50mm
        ),
        device=DeviceParams(
            diameter=256e-6,
            pixel_size=1e-6,
        ),
        target=TargetParams(
            target_type='size',
            target_span=(10e-3,),  # 10mm line
            tolerance=0.01,
            num_spots=(5, 1),
            splitter_mode='natural',
        ),
        optimization=OptimizationParams(
            phase_iters=800,
        ),
    )

    strategy = config.get_finite_distance_strategy()
    params = config.get_splitter_params()

    print(f"\nConfiguration:")
    print(f"  Working distance: {config.physical.working_distance*1000:.1f} mm")
    print(f"  Target size: {config.target.target_span[0]*1000:.1f} mm")
    print(f"  Strategy: {strategy.value if strategy else 'None'}")
    print(f"  Period: {params['period']*1e6:.2f} um")

    result = optimize_doe(config, device=device, verbose=True)

    print(f"\nResults:")
    print(f"  Uniformity: {result.metrics.order_uniformity:.4f}")
    print(f"  Total Efficiency: {result.metrics.total_efficiency:.4f}")
    print(f"  Has Fresnel phase: {result.fresnel_phase is not None}")

    return result


def test_tolerance_effect():
    """Test how tolerance affects period and efficiency."""
    print("\n" + "="*70)
    print("CASE 9: Tolerance effect on period size")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tolerances = [0.02, 0.01, 0.005]
    results = {}

    for tol in tolerances:
        print(f"\n--- Tolerance: {tol*100:.1f}% ---")

        config = DOEConfig(
            doe_type=DOEType.SPLITTER_2D,
            physical=PhysicalParams(
                wavelength=532e-9,
                refraction_index=1.46,
                working_distance=None,
            ),
            device=DeviceParams(
                diameter=256e-6,
                pixel_size=1e-6,
            ),
            target=TargetParams(
                target_type='angle',
                target_span=(0.15, 0.15),
                tolerance=tol,
                num_spots=(5, 5),
                splitter_mode='uniform',  # Use uniform mode to see tolerance effect
            ),
            optimization=OptimizationParams(
                phase_iters=600,
            ),
        )

        params = config.get_splitter_params()
        print(f"  Period: {params['period']*1e6:.2f} um")
        print(f"  Num orders (grid): {params['num_orders']}")

        result = optimize_doe(config, device=device, verbose=False)
        results[f"{tol*100:.1f}%"] = {
            'result': result,
            'period': params['period'],
            'num_orders': params['num_orders'],
        }

        print(f"  Uniformity: {result.metrics.order_uniformity:.4f}")
        print(f"  Efficiency: {result.metrics.total_efficiency:.4f}")

    return results


def test_pixel_multiplier_effect():
    """Test how pixel multiplier affects accuracy."""
    print("\n" + "="*70)
    print("CASE 10: Pixel multiplier effect on accuracy")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    multipliers = [1, 2]
    results = {}

    for mult in multipliers:
        print(f"\n--- Pixel Multiplier: {mult}x ---")

        config = DOEConfig(
            doe_type=DOEType.SPLITTER_2D,
            physical=PhysicalParams(
                wavelength=532e-9,
                refraction_index=1.46,
                working_distance=None,
            ),
            device=DeviceParams(
                diameter=256e-6,
                pixel_size=1e-6,
            ),
            target=TargetParams(
                target_type='angle',
                target_span=(0.2, 0.2),
                tolerance=0.01,
                num_spots=(5, 5),
                splitter_mode='natural',
            ),
            optimization=OptimizationParams(
                phase_iters=800,
                phase_pixel_multiplier=mult,
            ),
        )

        result = optimize_doe(config, device=device, verbose=False)
        results[f"{mult}x"] = result

        print(f"  Uniformity: {result.metrics.order_uniformity:.4f}")
        print(f"  Efficiency: {result.metrics.total_efficiency:.4f}")
        print(f"  Phase shape: {result.phase.shape}")

    return results


def test_upsample_factor():
    """Test different upsampling factors during optimization."""
    print("\n" + "="*70)
    print("CASE 11: Upsampling factor effect")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    factors = [1, 2]  # 1x and 2x upsampling
    results = {}

    for factor in factors:
        print(f"\n--- Upsample Factor: {factor}x ---")

        config = DOEConfig(
            doe_type=DOEType.SPLITTER_2D,
            physical=PhysicalParams(
                wavelength=532e-9,
                refraction_index=1.46,
                working_distance=None,
            ),
            device=DeviceParams(
                diameter=256e-6,
                pixel_size=1e-6,
            ),
            target=TargetParams(
                target_type='angle',
                target_span=(0.15, 0.15),
                tolerance=0.01,
                num_spots=(5, 5),
                splitter_mode='natural',
            ),
            optimization=OptimizationParams(
                phase_iters=800,
                simulation_upsample=factor,
            ),
        )

        result = optimize_doe(config, device=device, verbose=False)
        results[f"{factor}x"] = result

        print(f"  Uniformity: {result.metrics.order_uniformity:.4f}")
        print(f"  Total Efficiency: {result.metrics.total_efficiency:.4f}")

    return results


def analyze_efficiency():
    """Analyze why efficiency is below theoretical value."""
    print("\n" + "="*70)
    print("EFFICIENCY ANALYSIS")
    print("="*70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Simple 3x3 case for analysis
    config = DOEConfig(
        doe_type=DOEType.SPLITTER_2D,
        physical=PhysicalParams(
            wavelength=532e-9,
            refraction_index=1.46,
            working_distance=None,
        ),
        device=DeviceParams(
            diameter=256e-6,
            pixel_size=1e-6,
        ),
        target=TargetParams(
            target_type='angle',
            target_span=(0.1, 0.1),
            tolerance=0.01,
            num_spots=(3, 3),
            splitter_mode='natural',
        ),
        optimization=OptimizationParams(
            phase_iters=1000,
        ),
    )

    result = optimize_doe(config, device=device, verbose=True)

    n_spots = 9
    theoretical_per_spot = 1.0 / n_spots
    theoretical_total = 1.0  # Ideal: all energy goes to working orders

    print(f"\n--- Efficiency Analysis ---")
    print(f"  Number of spots: {n_spots}")
    print(f"  Theoretical per spot: {theoretical_per_spot:.4f}")
    print(f"  Actual mean per spot: {result.metrics.order_efficiency_mean:.4f}")
    print(f"  Ratio (actual/theoretical): {result.metrics.order_efficiency_mean / theoretical_per_spot:.4f}")
    print(f"  Total efficiency: {result.metrics.total_efficiency:.4f}")

    # Analyze where energy goes
    sim_intensity = result.simulated_intensity
    total_energy = sim_intensity.sum()
    print(f"\n--- Energy Distribution ---")
    print(f"  Total simulated intensity: {total_energy:.4f}")

    # Working orders energy
    params = config.get_splitter_params()
    working_energy = 0
    for py, px in params['order_positions']:
        working_energy += sim_intensity[py, px]
    print(f"  Working orders energy: {working_energy:.4f}")
    print(f"  Non-working orders energy: {total_energy - working_energy:.4f}")
    print(f"  Working order fraction: {working_energy / total_energy:.4f}")

    print(f"\n--- Loss Function Analysis ---")
    print(f"  Loss function normalizes both recon and target")
    print(f"  -> Optimizes SHAPE (relative efficiency), not ABSOLUTE efficiency")
    print(f"  -> Non-working order energy is not penalized directly")

    return result


def main():
    """Run all test cases and generate visualizations."""
    import os
    from doe_optimizer.utils import plot_1d_splitter_result

    # Create output directory
    output_dir = 'splitter_results'
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nMax optimization resolution: {MAX_OPTIMIZATION_RESOLUTION}")

    # Test Case 1: Infinite distance, odd spots
    result1 = test_infinite_distance_odd_spots()
    plot_comprehensive_splitter_result(
        result1,
        save_path=f'{output_dir}/case1_infinite_odd.png'
    )

    # Test Case 2: Infinite distance, even spots (symmetric)
    result2 = test_infinite_distance_even_spots()
    plot_comprehensive_splitter_result(
        result2,
        save_path=f'{output_dir}/case2_infinite_even.png'
    )

    # Test Case 3: Finite distance, large target (Strategy 2: Periodic + Fresnel)
    result3 = test_finite_distance_large_target()
    plot_comprehensive_splitter_result(
        result3,
        save_path=f'{output_dir}/case3_finite_large.png'
    )

    # Evaluate Case 3 with SFR propagation
    print("\n--- Evaluating Case 3 with SFR propagation ---")
    config3 = DOEConfig(
        doe_type=DOEType.SPLITTER_2D,
        physical=PhysicalParams(
            wavelength=532e-9,
            refraction_index=1.46,
            working_distance=100e-3,
        ),
        device=DeviceParams(
            diameter=256e-6,
            pixel_size=1e-6,
        ),
        target=TargetParams(
            target_type='size',
            target_span=(20e-3, 20e-3),
            tolerance=0.006,
            num_spots=(6, 6),
            splitter_mode='natural',
        ),
        optimization=OptimizationParams(),
    )
    eval3 = evaluate_finite_distance_splitter(
        result3.device_phase,
        config3,
        result3.splitter_params,
        use_fresnel_phase=True,
        fresnel_phase=result3.fresnel_phase,
        output_resolution=(512, 512),
        optimized_phase=result3.phase,  # Use original optimization-resolution phase for k-space evaluation
    )
    print(f"  SFR Evaluation:")
    print(f"    Total Efficiency: {eval3.total_efficiency:.4f}")
    print(f"    Mean Spot Efficiency: {eval3.mean_efficiency:.4f}")
    print(f"    Uniformity: {eval3.uniformity:.4f}")
    print(f"    Airy Radius: {eval3.airy_radius_meters*1e6:.1f} um")
    plot_finite_distance_evaluation(
        eval3,
        save_path=f'{output_dir}/case3_sfr_evaluation.png',
        title='Case 3: Strategy 2 (Periodic + Fresnel) SFR Evaluation'
    )

    # Test Case 3b: Finite distance, small target (Strategy 1: ASM)
    result3b = test_finite_distance_small_target()
    # Note: For ASM (Strategy 1), k-space visualization is not meaningful
    # since optimization is done directly in spatial domain with ASM propagation
    # The meaningful evaluation is done with ASM propagation below

    # Evaluate Case 3b with ASM propagation
    print("\n--- Evaluating Case 3b with ASM propagation ---")
    config3b = DOEConfig(
        doe_type=DOEType.SPLITTER_2D,
        physical=PhysicalParams(
            wavelength=532e-9,
            refraction_index=1.46,
            working_distance=1e-3,
        ),
        device=DeviceParams(
            diameter=256e-6,
            pixel_size=1e-6,
        ),
        target=TargetParams(
            target_type='size',
            target_span=(200e-6, 200e-6),  # Must match optimization
            tolerance=0.01,
            num_spots=(3, 3),
            splitter_mode='natural',
        ),
        optimization=OptimizationParams(),
    )
    eval3b = evaluate_finite_distance_splitter(
        result3b.device_phase,
        config3b,
        result3b.splitter_params,
        use_fresnel_phase=False,  # Strategy 1 does not use Fresnel overlay
    )
    print(f"  ASM Evaluation:")
    print(f"    Total Efficiency: {eval3b.total_efficiency:.4f}")
    print(f"    Mean Spot Efficiency: {eval3b.mean_efficiency:.4f}")
    print(f"    Uniformity: {eval3b.uniformity:.4f}")
    print(f"    Airy Radius: {eval3b.airy_radius_meters*1e6:.1f} um")
    plot_finite_distance_evaluation(
        eval3b,
        save_path=f'{output_dir}/case3b_asm_evaluation.png',
        title='Case 3b: Strategy 1 (ASM) Evaluation'
    )

    # Test Case 4: Natural vs Uniform comparison
    results4 = test_uniform_grid_comparison()

    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for idx, (mode, result) in enumerate(results4.items()):
        ax = axes[idx]
        eff = result.metrics.order_efficiencies
        params = result.splitter_params
        n_eff = len(eff)

        if params and 'working_orders' in params and len(params['working_orders']) == n_eff:
            labels = [f'({o[0]},{o[1]})' for o in params['working_orders']]
        else:
            labels = [str(i) for i in range(n_eff)]

        color = 'steelblue' if mode == 'natural' else 'forestgreen'
        ax.bar(range(n_eff), eff, color=color)

        # Theoretical line
        theoretical = 1.0 / n_eff
        ax.axhline(y=theoretical, color='green', linestyle=':', linewidth=2,
                   label=f'Theoretical: {theoretical:.4f}')
        ax.axhline(y=result.metrics.order_efficiency_mean, color='r', linestyle='--',
                   label=f'Mean: {result.metrics.order_efficiency_mean:.4f}')

        ax.set_xticks(range(n_eff))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel('Order (ny, nx)')
        ax.set_ylabel('Efficiency')
        ax.set_title(f'{mode.upper()} Grid ({n_eff} orders)\n'
                     f'Uniformity: {result.metrics.order_uniformity:.4f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/case4_comparison.png', dpi=150)
    print(f"\nComparison saved to {output_dir}/case4_comparison.png")

    # Test Case 5: 1D Splitter (use specialized 1D visualization)
    result5 = test_1d_splitter()
    plot_1d_splitter_result(
        result5,
        save_path=f'{output_dir}/case5_1d_splitter.png'
    )

    # Test Case 6: Large spot array (10x10)
    result6 = test_large_spot_array()
    plot_comprehensive_splitter_result(
        result6,
        save_path=f'{output_dir}/case6_large_array.png'
    )

    # Test Case 7: Different wavelengths comparison
    results7 = test_different_wavelengths()

    # Create wavelength comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    wavelength_names = list(results7.keys())

    for idx, (wl_name, result) in enumerate(results7.items()):
        ax = axes[idx]
        eff = result.metrics.order_efficiencies
        n_eff = len(eff)

        colors = {'405nm': 'purple', '532nm': 'green', '633nm': 'red'}
        ax.bar(range(n_eff), eff, color=colors.get(wl_name, 'steelblue'), alpha=0.7)

        theoretical = 1.0 / n_eff
        ax.axhline(y=theoretical, color='gray', linestyle=':', linewidth=2,
                   label=f'Theoretical: {theoretical:.4f}')
        ax.axhline(y=result.metrics.order_efficiency_mean, color='black', linestyle='--',
                   label=f'Mean: {result.metrics.order_efficiency_mean:.4f}')

        ax.set_xlabel('Order Index')
        ax.set_ylabel('Efficiency')
        ax.set_title(f'{wl_name}\n'
                     f'Uniformity: {result.metrics.order_uniformity:.4f}, '
                     f'Efficiency: {result.metrics.total_efficiency:.4f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/case7_wavelength_comparison.png', dpi=150)
    print(f"\nWavelength comparison saved to {output_dir}/case7_wavelength_comparison.png")

    # Test Case 8: 1D Finite distance
    result8 = test_1d_finite_distance()
    plot_1d_splitter_result(
        result8,
        save_path=f'{output_dir}/case8_1d_finite.png'
    )

    # Test Case 9: Tolerance effect
    results9 = test_tolerance_effect()

    # Create tolerance comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (tol_name, data) in enumerate(results9.items()):
        ax = axes[idx]
        result = data['result']
        eff = result.metrics.order_efficiencies
        n_eff = len(eff)

        ax.bar(range(n_eff), eff, color='steelblue', alpha=0.8)
        theoretical = 1.0 / n_eff
        ax.axhline(y=theoretical, color='green', linestyle=':', linewidth=2,
                   label=f'Theoretical: {theoretical:.4f}')
        ax.axhline(y=result.metrics.order_efficiency_mean, color='red', linestyle='--',
                   label=f'Mean: {result.metrics.order_efficiency_mean:.4f}')

        ax.set_xlabel('Order Index')
        ax.set_ylabel('Efficiency')
        ax.set_title(f'Tolerance: {tol_name}\n'
                     f'Period: {data["period"]*1e6:.1f} um\n'
                     f'Uniformity: {result.metrics.order_uniformity:.4f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/case9_tolerance_effect.png', dpi=150)
    print(f"\nTolerance effect saved to {output_dir}/case9_tolerance_effect.png")

    # Test Case 10: Pixel multiplier effect
    results10 = test_pixel_multiplier_effect()

    # Create pixel multiplier comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (mult_name, result) in enumerate(results10.items()):
        ax = axes[idx]
        eff = result.metrics.order_efficiencies
        n_eff = len(eff)

        ax.bar(range(n_eff), eff, color='forestgreen', alpha=0.8)
        theoretical = 1.0 / n_eff
        ax.axhline(y=theoretical, color='green', linestyle=':', linewidth=2,
                   label=f'Theoretical: {theoretical:.4f}')
        ax.axhline(y=result.metrics.order_efficiency_mean, color='red', linestyle='--',
                   label=f'Mean: {result.metrics.order_efficiency_mean:.4f}')

        ax.set_xlabel('Order Index')
        ax.set_ylabel('Efficiency')
        ax.set_title(f'Pixel Multiplier: {mult_name}\n'
                     f'Phase shape: {result.phase.shape}\n'
                     f'Uniformity: {result.metrics.order_uniformity:.4f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/case10_pixel_multiplier.png', dpi=150)
    print(f"\nPixel multiplier effect saved to {output_dir}/case10_pixel_multiplier.png")

    # Efficiency Analysis
    result_analysis = analyze_efficiency()
    plot_comprehensive_splitter_result(
        result_analysis,
        save_path=f'{output_dir}/efficiency_analysis.png'
    )

    # Test Case 11: Upsample factor comparison
    results11 = test_upsample_factor()

    # Create upsample factor comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for idx, (factor_name, result) in enumerate(results11.items()):
        ax = axes[idx]
        eff = result.metrics.order_efficiencies
        n_eff = len(eff)

        ax.bar(range(n_eff), eff, color='teal', alpha=0.8)
        theoretical = 1.0 / n_eff
        ax.axhline(y=theoretical, color='green', linestyle=':', linewidth=2,
                   label=f'Theoretical: {theoretical:.4f}')
        ax.axhline(y=result.metrics.order_efficiency_mean, color='red', linestyle='--',
                   label=f'Mean: {result.metrics.order_efficiency_mean:.4f}')

        ax.set_xlabel('Order Index')
        ax.set_ylabel('Efficiency')
        ax.set_title(f'Upsample Factor: {factor_name}\n'
                     f'Uniformity: {result.metrics.order_uniformity:.4f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/case11_upsample_factor.png', dpi=150)
    print(f"\nUpsample factor comparison saved to {output_dir}/case11_upsample_factor.png")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nCase 1 (Infinite, 5x5 odd, natural): Uniformity = {result1.metrics.order_uniformity:.4f}")
    print(f"Case 2 (Infinite, 4x4 even, natural): Uniformity = {result2.metrics.order_uniformity:.4f}")
    print(f"Case 3 (Finite 100mm, 6x6, Strategy 2):")
    print(f"    k-space Uniformity = {result3.metrics.order_uniformity:.4f}")
    print(f"    SFR Uniformity = {eval3.uniformity:.4f}")
    print(f"Case 3b (Finite 1mm, 3x3, Strategy 1 - ASM):")
    print(f"    ASM Uniformity = {eval3b.uniformity:.4f}")
    print(f"    ASM Total Efficiency = {eval3b.total_efficiency:.4f}")
    print(f"Case 4 Natural: Uniformity = {results4['natural'].metrics.order_uniformity:.4f}")
    print(f"Case 4 Uniform: Uniformity = {results4['uniform'].metrics.order_uniformity:.4f}")
    print(f"Case 5 (1D Splitter, 7 spots): Uniformity = {result5.metrics.order_uniformity:.4f}")
    print(f"Case 6 (10x10 array): Uniformity = {result6.metrics.order_uniformity:.4f}")
    print(f"Case 7 (Wavelength comparison):")
    for wl_name, result in results7.items():
        print(f"    {wl_name}: Uniformity = {result.metrics.order_uniformity:.4f}")
    print(f"Case 8 (1D Finite 50mm): Uniformity = {result8.metrics.order_uniformity:.4f}")
    print(f"Case 9 (Tolerance effect):")
    for tol_name, data in results9.items():
        print(f"    {tol_name}: Period = {data['period']*1e6:.1f} um, Uniformity = {data['result'].metrics.order_uniformity:.4f}")
    print(f"Case 10 (Pixel multiplier):")
    for mult_name, result in results10.items():
        print(f"    {mult_name}: Uniformity = {result.metrics.order_uniformity:.4f}")
    print(f"Case 11 (Upsample factor):")
    for factor_name, result in results11.items():
        print(f"    {factor_name}: Uniformity = {result.metrics.order_uniformity:.4f}")

    print(f"\nAll results saved to '{output_dir}/' directory")

    # plt.show()  # Commented out to avoid blocking in non-interactive mode


if __name__ == '__main__':
    main()
