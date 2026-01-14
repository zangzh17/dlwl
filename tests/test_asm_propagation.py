"""
Test ASM propagation with output_size parameter.

Tests:
1. Case A: output_size > DOE_size (padding)
2. Case B: output_size <= DOE_size (cropping)
3. Odd/even pixel count handling
4. Interface consistency with SFR
"""

import numpy as np
import torch
import sys
sys.path.insert(0, 'C:/Code/dlwl')

from doe_optimizer.core.propagation import propagation_ASM, propagation_SFR
from doe_optimizer.params.asm_params import ASMParams
from doe_optimizer.params.sfr_params import SFRParams
from doe_optimizer.params.base import PhysicalConstants
from doe_optimizer.core.propagator_factory import PropagatorBuilder


def test_asm_output_size():
    """Test ASM with various output_size configurations."""
    print("=" * 60)
    print("Testing ASM propagation with output_size parameter")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Common parameters
    wavelength = np.array([[[[532e-9]]]])
    z = np.array([[[[0.01]]]])  # 10mm
    pixel_size = 1e-6  # 1um

    # Test Case A: output_size > input_size (256um DOE -> 1mm target)
    print("\n--- Case A: output_size > input_size ---")
    doe_pixels = (256, 256)  # 256um x 256um DOE
    target_size = (0.001, 0.001)  # 1mm x 1mm target
    target_resolution = (256, 256)  # 256x256 output pixels

    # Create input field (simple aperture)
    field_a = torch.ones(1, 1, doe_pixels[0], doe_pixels[1], dtype=torch.complex64, device=device)
    field_a = field_a * torch.exp(1j * torch.rand_like(field_a.real) * 2 * np.pi)

    feature_size = (pixel_size, pixel_size)

    # Propagate
    out_a = propagation_ASM(
        field_a,
        feature_size=feature_size,
        wavelength=wavelength,
        z=z,
        output_size=target_size,
        output_resolution=target_resolution
    )

    print(f"  Input: {doe_pixels} pixels, {doe_pixels[0]*pixel_size*1e6:.0f}um x {doe_pixels[1]*pixel_size*1e6:.0f}um")
    print(f"  Output size: {target_size[0]*1e3:.1f}mm x {target_size[1]*1e3:.1f}mm")
    print(f"  Output resolution: {out_a.shape[-2:]} (expected {target_resolution})")
    assert out_a.shape[-2:] == target_resolution, f"Resolution mismatch: {out_a.shape[-2:]} != {target_resolution}"
    print("  PASS Case A passed")

    # Test Case B: output_size <= input_size (256um DOE -> 128um target)
    print("\n--- Case B: output_size <= input_size ---")
    target_size_b = (128e-6, 128e-6)  # 128um x 128um target
    target_resolution_b = (64, 64)  # 64x64 output pixels

    out_b = propagation_ASM(
        field_a,
        feature_size=feature_size,
        wavelength=wavelength,
        z=z,
        output_size=target_size_b,
        output_resolution=target_resolution_b
    )

    print(f"  Input: {doe_pixels} pixels, {doe_pixels[0]*pixel_size*1e6:.0f}um x {doe_pixels[1]*pixel_size*1e6:.0f}um")
    print(f"  Output size: {target_size_b[0]*1e6:.0f}um x {target_size_b[1]*1e6:.0f}um")
    print(f"  Output resolution: {out_b.shape[-2:]} (expected {target_resolution_b})")
    assert out_b.shape[-2:] == target_resolution_b, f"Resolution mismatch: {out_b.shape[-2:]} != {target_resolution_b}"
    print("  PASS Case B passed")

    # Test Case C: output_size = input_size (original behavior)
    print("\n--- Case C: output_size = input_size (original behavior) ---")
    target_size_c = (doe_pixels[0] * pixel_size, doe_pixels[1] * pixel_size)

    out_c = propagation_ASM(
        field_a,
        feature_size=feature_size,
        wavelength=wavelength,
        z=z,
        output_size=target_size_c,
        output_resolution=doe_pixels
    )

    print(f"  Output resolution: {out_c.shape[-2:]} (expected {doe_pixels})")
    assert out_c.shape[-2:] == doe_pixels, f"Resolution mismatch: {out_c.shape[-2:]} != {doe_pixels}"
    print("  PASS Case C passed")

    # Test Case D: None output_size (default to input size)
    print("\n--- Case D: output_size=None (default behavior) ---")
    out_d = propagation_ASM(
        field_a,
        feature_size=feature_size,
        wavelength=wavelength,
        z=z,
        output_size=None,
        output_resolution=None
    )

    print(f"  Output resolution: {out_d.shape[-2:]} (expected {doe_pixels})")
    assert out_d.shape[-2:] == doe_pixels, f"Resolution mismatch: {out_d.shape[-2:]} != {doe_pixels}"
    print("  PASS Case D passed")


def test_odd_even_pixels():
    """Test ASM with odd and even pixel counts."""
    print("\n" + "=" * 60)
    print("Testing odd/even pixel count handling")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wavelength = np.array([[[[532e-9]]]])
    z = np.array([[[[0.01]]]])
    pixel_size = 1e-6
    feature_size = (pixel_size, pixel_size)

    test_cases = [
        ((256, 256), (512, 512), (256, 256), "even->even->even"),
        ((255, 255), (510, 510), (255, 255), "odd->even->odd"),
        ((256, 256), (511, 511), (257, 257), "even->odd->odd"),
        ((255, 255), (512, 512), (256, 256), "odd->even->even"),
        ((100, 100), (300, 300), (150, 150), "even->even->even (arbitrary)"),
        ((101, 101), (303, 303), (151, 151), "odd->odd->odd (arbitrary)"),
    ]

    for doe_px, target_px_raw, out_res, name in test_cases:
        # Convert target_px to physical size
        target_size = (target_px_raw[0] * pixel_size, target_px_raw[1] * pixel_size)

        field = torch.ones(1, 1, doe_px[0], doe_px[1], dtype=torch.complex64, device=device)

        try:
            out = propagation_ASM(
                field,
                feature_size=feature_size,
                wavelength=wavelength,
                z=z,
                output_size=target_size,
                output_resolution=out_res
            )
            print(f"  {name}: DOE {doe_px} -> target {target_px_raw}px -> output {out.shape[-2:]} PASS")
            assert out.shape[-2:] == out_res, f"Expected {out_res}, got {out.shape[-2:]}"
        except Exception as e:
            print(f"  {name}: DOE {doe_px} -> target {target_px_raw}px FAILED: {e}")
            raise


def test_asm_params_interface():
    """Test ASMParams interface consistency with SFRParams."""
    print("\n" + "=" * 60)
    print("Testing ASMParams interface (should be similar to SFRParams)")
    print("=" * 60)

    physical = PhysicalConstants(
        wavelength=532e-9,
        refraction_index=1.62,
        pixel_size=1e-6
    )

    # Create ASMParams with new interface
    asm_params = ASMParams(
        doe_pixels=(256, 256),
        physical=physical,
        working_distances=[0.01],
        target_size=(0.001, 0.001),  # 1mm
        target_resolution=(256, 256)
    )

    print(f"  ASMParams:")
    print(f"    doe_pixels: {asm_params.doe_pixels}")
    print(f"    doe_size: {asm_params.doe_size}")
    print(f"    target_size: {asm_params.target_size}")
    print(f"    target_resolution: {asm_params.target_resolution}")
    print(f"    output_pixels: {asm_params.output_pixels}")

    # Create equivalent SFRParams for comparison
    sfr_params = SFRParams(
        doe_pixels=(256, 256),
        physical=physical,
        working_distances=[0.01],
        target_size=(0.001, 0.001),
        target_resolution=(256, 256)
    )

    print(f"\n  SFRParams (for comparison):")
    print(f"    doe_pixels: {sfr_params.doe_pixels}")
    print(f"    doe_size: {sfr_params.doe_size}")
    print(f"    target_size: {sfr_params.target_size}")
    print(f"    target_resolution: {sfr_params.target_resolution}")

    # Check PropagatorConfig
    asm_config = asm_params.to_propagator_config()
    sfr_config = sfr_params.to_propagator_config()

    print(f"\n  PropagatorConfig comparison:")
    print(f"    ASM output_size: {asm_config.output_size}")
    print(f"    SFR output_size: {sfr_config.output_size}")
    print(f"    ASM output_resolution: {asm_config.output_resolution}")
    print(f"    SFR output_resolution: {sfr_config.output_resolution}")

    assert asm_config.output_size == sfr_config.output_size, "output_size mismatch"
    assert asm_config.output_resolution == sfr_config.output_resolution, "output_resolution mismatch"
    print("\n  PASS Interface consistency test passed")


def test_propagator_builder():
    """Test PropagatorBuilder with ASM."""
    print("\n" + "=" * 60)
    print("Testing PropagatorBuilder with ASM")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    physical = PhysicalConstants(
        wavelength=532e-9,
        refraction_index=1.62,
        pixel_size=1e-6
    )

    # Create ASMParams
    asm_params = ASMParams(
        doe_pixels=(256, 256),
        physical=physical,
        working_distances=[0.01],
        target_size=(0.001, 0.001),  # 1mm output
        target_resolution=(256, 256)
    )

    # Build propagator
    config = asm_params.to_propagator_config()
    builder = PropagatorBuilder(config, device=device)
    propagator = builder.build()

    # Create input field
    field = torch.ones(1, 1, 256, 256, dtype=torch.complex64, device=device)

    # Propagate
    output = propagator(field)

    print(f"  Input shape: {field.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Expected output: (1, 1, 256, 256)")

    assert output.shape == (1, 1, 256, 256), f"Unexpected output shape: {output.shape}"
    print("  PASS PropagatorBuilder test passed")


def test_energy_conservation():
    """Test that energy is approximately conserved."""
    print("\n" + "=" * 60)
    print("Testing energy conservation")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    wavelength = np.array([[[[532e-9]]]])
    z = np.array([[[[0.01]]]])
    pixel_size = 1e-6
    feature_size = (pixel_size, pixel_size)

    # Create input with known energy
    doe_pixels = (256, 256)
    field = torch.ones(1, 1, *doe_pixels, dtype=torch.complex64, device=device)
    input_energy = (field.abs() ** 2).sum().item()

    # Test Case A: output > input
    target_size_a = (0.001, 0.001)
    out_a = propagation_ASM(
        field,
        feature_size=feature_size,
        wavelength=wavelength,
        z=z,
        output_size=target_size_a,
        output_resolution=(256, 256)
    )
    output_energy_a = (out_a.abs() ** 2).sum().item()

    # Test Case B: output < input
    target_size_b = (128e-6, 128e-6)
    out_b = propagation_ASM(
        field,
        feature_size=feature_size,
        wavelength=wavelength,
        z=z,
        output_size=target_size_b,
        output_resolution=(128, 128)
    )
    output_energy_b = (out_b.abs() ** 2).sum().item()

    print(f"  Input energy: {input_energy:.4f}")
    print(f"  Case A (output > input) energy: {output_energy_a:.4f}")
    print(f"  Case B (output < input) energy: {output_energy_b:.4f}")
    print(f"  Note: Energy may not be exactly conserved due to:")
    print(f"        - Cropping in Case B removes energy")
    print(f"        - Resampling can affect energy")
    print("  PASS Energy test complete (informational only)")


if __name__ == "__main__":
    try:
        test_asm_output_size()
        test_odd_even_pixels()
        test_asm_params_interface()
        test_propagator_builder()
        test_energy_conservation()
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
