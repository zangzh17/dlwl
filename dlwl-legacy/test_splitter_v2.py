"""Test beam splitter optimization with v2.0 API.

Standardized test suite for all propagation types:
- Type A (FFT): k-space / infinite distance
- Type B (SFR): Large target, finite distance, Strategy 1
- Type C (ASM): Small target, finite distance, Strategy 1
- Strategy 2 (Periodic+Fresnel): Finite distance with periodic optimization

Each test case includes:
- Complete standardized visualization
- 1x and 2x upsampling evaluation with efficiency analysis
- Structured parameter display
- Parameter conversion details
- **Actual angles (degrees) for all order displays**

Also includes upsampling during optimization examples.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
import math

from doe_optimizer import (
    OptimizationRunner,
    run_optimization,
    ProgressInfo,
)
from doe_optimizer.wizard.splitter import SplitterWizard, FiniteDistanceStrategy
from doe_optimizer.params.base import PropagationType


# Global loss history for tracking
loss_history = []


def print_progress(info: ProgressInfo):
    """Print progress callback with scientific notation."""
    global loss_history
    if info.current_iter == 0:
        loss_history.clear()
    loss_history.append((info.current_iter, info.current_loss))

    if info.current_iter % 100 == 0 or info.current_iter == info.total_iters - 1:
        print(f"  [{info.stage}] Iter {info.current_iter}/{info.total_iters}, Loss: {info.current_loss:.2e}")


def get_loss_history():
    """Get a copy of loss history."""
    return list(loss_history)


def order_to_angle(order_m, wavelength, period):
    """Convert diffraction order to angle in radians.

    sin(θ) = m * λ / Λ

    Args:
        order_m: Order number (integer or float)
        wavelength: Wavelength in meters
        period: Grating period in meters

    Returns:
        Angle in radians, or None if invalid
    """
    sin_theta = order_m * wavelength / period
    if abs(sin_theta) > 1.0:
        return None  # Evanescent
    return math.asin(sin_theta)


def orders_to_angles_deg(orders, wavelength, period):
    """Convert list of orders to angles in degrees.

    Args:
        orders: List of order numbers
        wavelength: Wavelength in meters
        period: Grating period in meters

    Returns:
        List of angles in degrees
    """
    angles = []
    for m in orders:
        theta = order_to_angle(m, wavelength, period)
        if theta is not None:
            angles.append(math.degrees(theta))
        else:
            angles.append(float('nan'))
    return angles


def evaluate_with_upsampling(phase, upsample_factor=2, crop_to_original=False):
    """Re-evaluate phase with higher upsampling using FFT.

    Args:
        phase: Optimized phase array (2D)
        upsample_factor: Upsampling factor for evaluation
        crop_to_original: If True, crop the output to original size (same angular range)
                         If False, return full upsampled k-space (denser angular sampling)

    Returns:
        Simulated intensity array (cropped or full based on crop_to_original)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    phase_t = torch.tensor(phase, device=device, dtype=torch.float32)
    if len(phase_t.shape) == 2:
        phase_t = phase_t.unsqueeze(0).unsqueeze(0)

    h, w = phase_t.shape[-2:]

    # Detect if this is effectively 1D (one dimension is 1)
    is_1d = (w == 1 or h == 1)

    if is_1d:
        # For 1D-like arrays, only upsample the non-trivial dimension
        # This prevents (N, 1) -> (2N, 2) which causes 16x energy instead of 4x
        if w == 1:
            new_h, new_w = h * upsample_factor, 1
        else:
            new_h, new_w = 1, w * upsample_factor
    else:
        new_h, new_w = h * upsample_factor, w * upsample_factor

    # Upsample phase using bilinear interpolation
    phase_up = torch.nn.functional.interpolate(
        phase_t, size=(new_h, new_w), mode='bilinear', align_corners=False
    )

    # Create field and propagate (FFT)
    field = torch.exp(1j * phase_up.to(torch.complex64))
    output_field = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field)))
    output_intensity = (output_field.abs() ** 2).squeeze().cpu().numpy()

    if crop_to_original:
        # Crop center region to get same angular range as original
        # The center h×w region of the 2h×2w FFT corresponds to the same angles
        if len(output_intensity.shape) == 2:
            cy, cx = new_h // 2, new_w // 2
            start_y, start_x = cy - h // 2, cx - w // 2
            output_intensity = output_intensity[start_y:start_y+h, start_x:start_x+w]
        else:
            # 1D case
            c = new_h // 2
            start = c - h // 2
            output_intensity = output_intensity[start:start+h]

        # Normalize to match 1x energy scale
        # For 2D FFT, energy scales as (N*M)^2, so:
        # - 1D upsample (N,1 -> 2N,1): energy scales as (2N)^2/(N)^2 = 4x
        # - 2D upsample (N,M -> 2N,2M): energy scales as (2N*2M)^2/(N*M)^2 = 16x
        if is_1d:
            # Only one dimension was upsampled, so factor is upsample_factor^2
            output_intensity = output_intensity / (upsample_factor ** 2)
        else:
            # Both dimensions were upsampled, factor is upsample_factor^4
            output_intensity = output_intensity / (upsample_factor ** 4)

    return output_intensity


def evaluate_physical_with_upsampling(phase, wavelength, pixel_size, working_distance,
                                       upsample_factor=2, propagation_type='asm',
                                       output_pixels=None, target_physical_size=None):
    """Re-evaluate phase with upsampling for physical (ASM/SFR) propagation.

    The key insight is that we want the SAME physical output plane as 1x,
    so we need to pass proper output_resolution and output_size parameters.

    IMPORTANT: For ASM/SFR, the phase must be upsampled correctly to preserve
    the wavefront. We use FFT-based upsampling (sinc interpolation) on the
    complex field, not bilinear on the phase.

    Args:
        phase: Optimized phase array (2D)
        wavelength: Wavelength in meters
        pixel_size: DOE pixel size in meters
        working_distance: Propagation distance in meters
        upsample_factor: Upsampling factor
        propagation_type: 'asm' or 'sfr'
        output_pixels: Original output resolution (H, W) from 1x - REQUIRED for correct behavior
        target_physical_size: Physical target size in meters (for SFR) - tuple (H, W)

    Returns:
        Simulated intensity array matching original output size
    """
    from doe_optimizer.core.propagation import propagation_ASM, propagation_SFR
    import numpy as np

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    phase_t = torch.tensor(phase, device=device, dtype=torch.float32)
    if len(phase_t.shape) == 2:
        phase_t = phase_t.unsqueeze(0).unsqueeze(0)

    h, w = phase_t.shape[-2:]
    new_h, new_w = h * upsample_factor, w * upsample_factor

    # Create complex field from original phase
    field_1x = torch.exp(1j * phase_t.to(torch.complex64))

    # Upsample complex field using FFT-based interpolation (sinc interpolation)
    # This preserves the wavefront structure correctly
    # FFT -> zero-pad in frequency domain -> IFFT
    F_field = torch.fft.fftshift(torch.fft.fft2(field_1x), dim=(-2, -1))

    # Zero-pad to new size
    pad_h = (new_h - h) // 2
    pad_w = (new_w - w) // 2
    F_padded = torch.nn.functional.pad(
        F_field, (pad_w, pad_w, pad_h, pad_h), mode='constant', value=0
    )

    # Inverse FFT with proper scaling
    # For FFT upsampling: zero-pad NxN to (2N)x(2N) spectrum
    # Scale factor = upsample_factor (not upsample_factor^2) preserves intensity
    # Verified empirically: scale=2 gives intensity ratio ~0.97 vs 1x
    scale = upsample_factor
    field = torch.fft.ifft2(torch.fft.ifftshift(F_padded, dim=(-2, -1))) * scale

    # Propagate with half pixel size
    upsampled_pixel_size = pixel_size / upsample_factor
    feature_size = (upsampled_pixel_size, upsampled_pixel_size)

    wavelength_arr = np.array([[[[wavelength]]]])
    z_arr = np.array([[[[working_distance]]]])

    # Use SAME output resolution as 1x to get same physical output plane
    # This is the key fix - we need the output to represent the same physical target
    if output_pixels is None:
        output_pixels = (h, w)  # Fallback to DOE size

    if propagation_type == 'sfr' and target_physical_size is not None:
        # SFR: use zoom-FFT to output to different physical plane
        output_field = propagation_SFR(
            field,
            feature_size=feature_size,
            wavelength=wavelength_arr,
            z=z_arr,
            output_size=target_physical_size,
            output_resolution=output_pixels
        )
    else:
        # ASM: output plane has same pixel size as DOE (after linear_conv crop)
        # Use output_resolution to interpolate to target size
        output_field = propagation_ASM(
            field,
            feature_size=feature_size,
            wavelength=wavelength_arr,
            z=z_arr,
            output_resolution=output_pixels,
            linear_conv=True
        )

    output_intensity = (output_field.abs() ** 2).squeeze().cpu().numpy()

    return output_intensity


def evaluate_asm_with_upsampling(phase, wavelength, pixel_size, working_distance,
                                  output_pixels, upsample_factor=2):
    """Re-evaluate phase with ASM propagation at higher resolution.

    Args:
        phase: Optimized phase array (2D)
        wavelength: Wavelength in meters
        pixel_size: DOE pixel size in meters
        working_distance: Propagation distance in meters
        output_pixels: Target output size (h, w)
        upsample_factor: Upsampling factor for evaluation

    Returns:
        Upsampled simulated intensity array
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    phase_t = torch.tensor(phase, device=device, dtype=torch.float32)
    if len(phase_t.shape) == 2:
        phase_t = phase_t.unsqueeze(0).unsqueeze(0)

    h, w = phase_t.shape[-2:]
    new_h, new_w = h * upsample_factor, w * upsample_factor

    # Upsample phase using bilinear interpolation
    phase_up = torch.nn.functional.interpolate(
        phase_t, size=(new_h, new_w), mode='bilinear', align_corners=False
    )

    # Create ASM propagator with upsampled pixel size
    upsampled_pixel_size = pixel_size / upsample_factor
    upsampled_output_pixels = (output_pixels[0] * upsample_factor, output_pixels[1] * upsample_factor)

    propagator = ASMPropagator(
        wavelength=wavelength,
        pixel_size=upsampled_pixel_size,
        distance=working_distance,
        input_shape=phase_up.shape,
        output_pixels=upsampled_output_pixels,
        device=device
    )

    # Propagate
    field = torch.exp(1j * phase_up.to(torch.complex64))
    output_field = propagator(field)
    output_intensity = (output_field.abs() ** 2).squeeze().cpu().numpy()

    return output_intensity


def compute_efficiency_with_airy(intensity, order_positions, airy_radius_pixels):
    """Compute efficiency using Airy disk integration.

    Args:
        intensity: 2D intensity array
        order_positions: List of (py, px) pixel positions
        airy_radius_pixels: Airy disk radius in pixels

    Returns:
        (total_efficiency, order_efficiencies, uniformity)
    """
    h = intensity.shape[0]
    w = intensity.shape[1] if len(intensity.shape) > 1 else 1

    total_intensity = intensity.sum()
    if total_intensity < 1e-10:
        return 0.0, [], 0.0

    order_efficiencies = []
    radius = int(airy_radius_pixels)

    for cy, cx in order_positions:
        if len(intensity.shape) == 1:
            # 1D case
            start = max(0, cy - radius)
            end = min(h, cy + radius + 1)
            disk_sum = float(intensity[start:end].sum())
        else:
            # 2D case: sum over circular disk
            y_min = max(0, cy - radius)
            y_max = min(h, cy + radius + 1)
            x_min = max(0, cx - radius)
            x_max = min(w, cx + radius + 1)

            disk_sum = 0.0
            for y in range(y_min, y_max):
                for x in range(x_min, x_max):
                    if (y - cy) ** 2 + (x - cx) ** 2 <= radius ** 2:
                        disk_sum += float(intensity[y, x])

        eff = disk_sum / float(total_intensity)
        order_efficiencies.append(eff)

    total_efficiency = sum(order_efficiencies)

    if order_efficiencies and max(order_efficiencies) + min(order_efficiencies) > 1e-10:
        uniformity = 1.0 - (max(order_efficiencies) - min(order_efficiencies)) / (
            max(order_efficiencies) + min(order_efficiencies)
        )
    else:
        uniformity = 0.0

    return total_efficiency, order_efficiencies, uniformity


def compute_efficiency_from_intensity(intensity, order_positions, integration_radius=1):
    """Compute efficiency from intensity at order positions.

    Args:
        intensity: 2D intensity array
        order_positions: List of (py, px) pixel positions
        integration_radius: Radius for integration (in pixels)

    Returns:
        (total_efficiency, order_efficiencies, uniformity)
    """
    h = intensity.shape[0]
    w = intensity.shape[1] if len(intensity.shape) > 1 else 1

    total_intensity = intensity.sum()
    if total_intensity < 1e-10:
        return 0.0, [], 0.0

    order_efficiencies = []
    for py, px in order_positions:
        # Integrate over small region
        y_min = max(0, py - integration_radius)
        y_max = min(h, py + integration_radius + 1)
        x_min = max(0, px - integration_radius)
        x_max = min(w, px + integration_radius + 1)

        if len(intensity.shape) > 1:
            region_sum = intensity[y_min:y_max, x_min:x_max].sum()
        else:
            region_sum = intensity[y_min:y_max].sum()

        eff = float(region_sum) / float(total_intensity)
        order_efficiencies.append(eff)

    total_efficiency = sum(order_efficiencies)

    if order_efficiencies and max(order_efficiencies) + min(order_efficiencies) > 1e-10:
        uniformity = 1.0 - (max(order_efficiencies) - min(order_efficiencies)) / (
            max(order_efficiencies) + min(order_efficiencies)
        )
    else:
        uniformity = 0.0

    return total_efficiency, order_efficiencies, uniformity


def scale_order_positions(order_positions, upsample_factor, original_size):
    """Scale order positions for upsampled array.

    Args:
        order_positions: Original order positions [(py, px), ...]
        upsample_factor: Upsampling factor
        original_size: (h, w) of original array

    Returns:
        Scaled order positions
    """
    h_orig, w_orig = original_size
    h_new = h_orig * upsample_factor
    w_new = w_orig * upsample_factor

    scaled = []
    for py, px in order_positions:
        # Scale relative to center
        cy_orig, cx_orig = h_orig // 2, w_orig // 2
        cy_new, cx_new = h_new // 2, w_new // 2

        dy = py - cy_orig
        dx = px - cx_orig

        new_py = cy_new + dy * upsample_factor
        new_px = cx_new + dx * upsample_factor

        scaled.append((int(new_py), int(new_px)))

    return scaled


def compute_airy_radius(wavelength, working_distance, doe_diameter):
    """Compute Airy disk radius.

    R_airy = 1.22 * λ * z / D

    Args:
        wavelength: Wavelength in meters
        working_distance: Working distance in meters
        doe_diameter: DOE diameter in meters

    Returns:
        Airy radius in meters
    """
    return 1.22 * wavelength * working_distance / doe_diameter


def format_structured_params(result):
    """Format structured parameters for display."""
    splitter_info = result.get('splitter_info', {})
    computed = result.get('computed_params', {})

    lines = []

    # Strategy
    strategy = splitter_info.get('strategy', None)
    lines.append(f"Strategy: {strategy if strategy else 'FFT (k-space)'}")

    # Grid mode
    grid_mode = splitter_info.get('grid_mode', 'natural')
    lines.append(f"Grid Mode: {grid_mode}")

    # Num spots
    num_spots = splitter_info.get('num_spots', 'N/A')
    lines.append(f"Num Spots: {num_spots}")

    # Period
    period_pixels = splitter_info.get('period_pixels', computed.get('period_pixels', 'N/A'))
    period = computed.get('period', None)
    if period:
        lines.append(f"Period: {period*1e6:.2f}μm ({period_pixels}px)")
    else:
        lines.append(f"Period: {period_pixels}px")

    # Target span
    theta_span = splitter_info.get('theta_span', None)
    target_span = splitter_info.get('target_span', None)
    if target_span:
        lines.append(f"Target Span: {target_span[0]*1e3:.2f}mm")
    elif theta_span:
        lines.append(f"Angle Span: {math.degrees(theta_span[0]):.2f}°")

    # Margin factor
    margin = splitter_info.get('target_margin_factor', None)
    if margin:
        lines.append(f"Margin Factor: {margin:.2f}")

    # Tolerance limit
    tol_limit = computed.get('tolerance_limit', None)
    if tol_limit:
        lines.append(f"Tolerance Limit: {tol_limit*100:.2f}%")

    return '\n'.join(lines)


def visualize_2d_standard(result, title, wavelength, pixel_size,
                           working_distance=None, target_span=None,
                           target_pixel_size=None, is_physical=False):
    """Standardized 2D visualization with complete analysis.

    Includes:
    - Row 1: Phase, Simulated 1x (linear), Simulated 2x (linear), Loss curve
    - Row 2: Efficiency 1x, Efficiency 2x, Order positions 1x (angles), Order positions 2x (angles)
    - Row 3: Structured parameters and metrics summary

    For FFT (k-space): Order positions shown as diffraction angles
    For ASM/SFR (physical): Positions shown in mm with Airy-based efficiency
    """
    phase = np.array(result['phase'])
    target = np.array(result['target_intensity'])
    simulated = np.array(result['simulated_intensity'])
    metrics = result.get('metrics', {})
    splitter_info = result.get('splitter_info', {})
    current_loss_history = get_loss_history()

    h_phase, w_phase = phase.shape
    h_sim, w_sim = simulated.shape

    # Get period for angle conversion
    period_pixels = splitter_info.get('period_pixels', h_phase)
    period_m = period_pixels * pixel_size

    # Get order positions
    order_positions = splitter_info.get('order_positions', [])
    order_efficiencies = metrics.get('order_efficiencies', [])
    working_orders = splitter_info.get('working_orders', [])

    # For ASM/SFR, compute Airy radius and use proper ASM upsampling evaluation
    if is_physical and working_distance:
        doe_diameter = h_phase * pixel_size
        airy_radius_m = compute_airy_radius(wavelength, working_distance, doe_diameter)
        target_px_size = target_pixel_size if target_pixel_size else pixel_size
        airy_radius_px = max(3, int(airy_radius_m / target_px_size))

        # Re-compute 1x efficiency with Airy integration
        if order_positions and simulated.size > 0:
            _, order_efficiencies, uni_1x = compute_efficiency_with_airy(
                simulated, order_positions, airy_radius_px
            )

        # Determine propagation type from strategy
        strategy = splitter_info.get('strategy', 'asm')
        prop_type = 'sfr' if strategy == 'sfr' else 'asm'

        # Compute target physical size for SFR
        if target_span is not None:
            margin = splitter_info.get('target_margin_factor', 1.1)
            target_physical_size = (target_span * margin, target_span * margin)
        else:
            target_physical_size = None

        # Compute 2x upsampled result using proper ASM/SFR propagation
        # Key: pass output_pixels=(h_sim, w_sim) so output represents same physical plane
        try:
            sim_2x = evaluate_physical_with_upsampling(
                phase, wavelength, pixel_size, working_distance,
                upsample_factor=2,
                propagation_type=prop_type,
                output_pixels=(h_sim, w_sim),
                target_physical_size=target_physical_size
            )
            h_2x, w_2x = sim_2x.shape

            # 2x efficiency uses same positions (physical space unchanged)
            if order_positions:
                _, eff_2x, uni_2x = compute_efficiency_with_airy(
                    sim_2x, order_positions, airy_radius_px
                )
            else:
                eff_2x = []
                uni_2x = 0.0
            order_positions_2x = order_positions  # Same physical positions
        except Exception as e:
            print(f"  Warning: {prop_type.upper()} 2x upsampling failed: {e}")
            import traceback
            traceback.print_exc()
            sim_2x = None
            eff_2x = []
            uni_2x = 0.0
            order_positions_2x = []
            h_2x, w_2x = 0, 0
    else:
        airy_radius_px = None
        # Compute 2x upsampled result for FFT with crop to same angular range
        sim_2x = evaluate_with_upsampling(phase, upsample_factor=2, crop_to_original=True)
        h_2x, w_2x = sim_2x.shape  # Same as original since cropped

        # Compute 2x efficiency at same positions (same angular range)
        # For FFT k-space, use integration_radius=0 to avoid overlap at consecutive orders
        if order_positions:
            # Since we cropped to original size, positions are the same
            _, eff_2x, uni_2x = compute_efficiency_from_intensity(
                sim_2x, order_positions, integration_radius=0
            )
            order_positions_2x = order_positions
        else:
            order_positions_2x = []
            eff_2x = []
            uni_2x = 0.0

    # Compute angle-based positions for FFT visualization
    if not is_physical and working_orders:
        # Convert orders to angles
        orders_y = [o[0] for o in working_orders]
        orders_x = [o[1] for o in working_orders]
        angles_y = orders_to_angles_deg(orders_y, wavelength, period_m)
        angles_x = orders_to_angles_deg(orders_x, wavelength, period_m)
    else:
        angles_y, angles_x = [], []

    # Compute extents
    if is_physical:
        # Physical space (ASM/SFR)
        phase_extent = [0, w_phase * pixel_size * 1e6, 0, h_phase * pixel_size * 1e6]
        target_px_size = target_pixel_size if target_pixel_size else pixel_size
        sim_extent = [
            -w_sim/2 * target_px_size * 1e3, w_sim/2 * target_px_size * 1e3,
            -h_sim/2 * target_px_size * 1e3, h_sim/2 * target_px_size * 1e3
        ]
        sim_2x_extent = sim_extent  # Same physical extent for ASM
        x_label = 'X (mm)'
        y_label = 'Y (mm)'
    else:
        # K-space (FFT) - compute angle extents
        phase_extent = [0, w_phase * pixel_size * 1e6, 0, h_phase * pixel_size * 1e6]
        # Max angle at edge of k-space
        max_order = h_sim // 2
        max_angle = order_to_angle(max_order, wavelength, period_m)
        if max_angle:
            max_angle_deg = math.degrees(max_angle)
            sim_extent = [-max_angle_deg, max_angle_deg, -max_angle_deg, max_angle_deg]
        else:
            sim_extent = [-h_sim//2, h_sim//2, -h_sim//2, h_sim//2]
        # For cropped 2x, same extent as 1x
        sim_2x_extent = sim_extent
        x_label = 'Angle θx (°)'
        y_label = 'Angle θy (°)'

    # Create figure
    fig = plt.figure(figsize=(20, 15))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 0.6])

    # ===== Row 1: Phase, Simulated 1x, Simulated 2x, Loss =====

    # Phase
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(phase, cmap='twilight', vmin=0, vmax=2*np.pi,
                   extent=phase_extent, origin='lower')
    ax.set_title('Optimized Phase (rad)')
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    plt.colorbar(im, ax=ax)

    # Simulated 1x (LINEAR scale)
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(simulated, cmap='hot', extent=sim_extent, origin='lower')
    ax.set_title('Simulated 1x (linear)')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.colorbar(im, ax=ax)

    # Simulated 2x (LINEAR scale) - same extent since cropped/matched
    ax = fig.add_subplot(gs[0, 2])
    if sim_2x is not None:
        im = ax.imshow(sim_2x, cmap='hot', extent=sim_2x_extent, origin='lower')
        ax.set_title('Simulated 2x (linear, cropped)')
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.colorbar(im, ax=ax)
    else:
        ax.text(0.5, 0.5, '2x upsampling\nfailed',
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title('Simulated 2x (failed)')
        ax.axis('off')

    # Loss curve
    ax = fig.add_subplot(gs[0, 3])
    if current_loss_history:
        iters = [x[0] for x in current_loss_history]
        losses = [x[1] for x in current_loss_history]
        ax.semilogy(iters, losses, 'b-', linewidth=1.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
    ax.set_title('Loss Curve')

    # ===== Row 2: Efficiency 1x, Efficiency 2x, Order pos 1x, Order pos 2x =====

    # Efficiency 1x
    ax = fig.add_subplot(gs[1, 0])
    if order_efficiencies:
        n = len(order_efficiencies)
        ax.bar(range(n), order_efficiencies, color='steelblue', alpha=0.8)
        theoretical = 1.0 / n
        ax.axhline(y=theoretical, color='green', linestyle=':', label=f'Theory: {theoretical:.4f}')
        ax.axhline(y=np.mean(order_efficiencies), color='red', linestyle='--',
                   label=f'Mean: {np.mean(order_efficiencies):.4f}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    total_eff = metrics.get('total_efficiency', sum(order_efficiencies) if order_efficiencies else 0)
    uni_1x = metrics.get('uniformity', 0)
    airy_note = f" (Airy r={airy_radius_px}px)" if airy_radius_px else ""
    ax.set_title(f'Efficiency 1x{airy_note} (Total={total_eff:.3f}, Uni={uni_1x:.3f})')
    ax.set_xlabel('Order Index')
    ax.set_ylabel('Efficiency')

    # Efficiency 2x
    ax = fig.add_subplot(gs[1, 1])
    if eff_2x:
        n = len(eff_2x)
        ax.bar(range(n), eff_2x, color='forestgreen', alpha=0.8)
        theoretical = 1.0 / n
        ax.axhline(y=theoretical, color='blue', linestyle=':', label=f'Theory: {theoretical:.4f}')
        ax.axhline(y=np.mean(eff_2x), color='red', linestyle='--',
                   label=f'Mean: {np.mean(eff_2x):.4f}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    total_eff_2x = sum(eff_2x) if eff_2x else 0
    ax.set_title(f'Efficiency 2x (Total={total_eff_2x:.3f}, Uni={uni_2x:.3f})')
    ax.set_xlabel('Order Index')
    ax.set_ylabel('Efficiency')

    # Order positions 1x - use ANGLES for FFT, mm for physical
    ax = fig.add_subplot(gs[1, 2])
    if order_positions and order_efficiencies:
        if is_physical:
            target_px_size = target_pixel_size if target_pixel_size else pixel_size
            pos_y = [(p[0] - h_sim/2) * target_px_size * 1e3 for p in order_positions]
            pos_x = [(p[1] - w_sim/2) * target_px_size * 1e3 for p in order_positions]
        else:
            # Use actual angles
            pos_y = angles_y
            pos_x = angles_x
        sc = ax.scatter(pos_x, pos_y, c=order_efficiencies, cmap='viridis',
                       s=100, edgecolors='black')
        plt.colorbar(sc, ax=ax, label='Efficiency')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    ax.set_title('Order Positions 1x')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Order positions 2x
    ax = fig.add_subplot(gs[1, 3])
    if order_positions_2x and eff_2x:
        if is_physical:
            target_px_size = target_pixel_size if target_pixel_size else pixel_size
            # For 2x upsampled physical space, pixel size is halved
            pos_y = [(p[0] - h_2x/2) * target_px_size/2 * 1e3 for p in order_positions_2x]
            pos_x = [(p[1] - w_2x/2) * target_px_size/2 * 1e3 for p in order_positions_2x]
        else:
            # For 2x upsampled FFT, angles are the same (just denser sampling)
            pos_y = angles_y
            pos_x = angles_x
        sc = ax.scatter(pos_x, pos_y, c=eff_2x, cmap='viridis',
                       s=100, edgecolors='black')
        plt.colorbar(sc, ax=ax, label='Efficiency')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    ax.set_title('Order Positions 2x')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # ===== Row 3: Parameters and summary =====

    # Structured parameters
    ax = fig.add_subplot(gs[2, :2])
    ax.axis('off')
    param_text = format_structured_params(result)
    summary_text = f"""
STRUCTURED PARAMETERS:
{param_text}

PHYSICAL PARAMETERS:
- Wavelength: {wavelength*1e9:.1f} nm
- Pixel Size: {pixel_size*1e6:.2f} μm
- DOE Size: {h_phase * pixel_size * 1e6:.0f} μm
"""
    if working_distance:
        summary_text += f"- Working Distance: {working_distance*1e3:.2f} mm\n"
    if target_span:
        summary_text += f"- Target Span: {target_span*1e3:.2f} mm\n"

    ax.text(0.02, 0.95, summary_text.strip(), fontsize=9, verticalalignment='top',
            fontfamily='monospace', transform=ax.transAxes)

    # Metrics summary
    ax = fig.add_subplot(gs[2, 2:])
    ax.axis('off')
    metrics_text = f"""
METRICS COMPARISON:

               1x Upsample    2x Upsample
Total Eff:     {total_eff:.4f}         {total_eff_2x:.4f}
Uniformity:    {uni_1x:.4f}         {uni_2x:.4f}
Mean Eff:      {np.mean(order_efficiencies) if order_efficiencies else 0:.6f}    {np.mean(eff_2x) if eff_2x else 0:.6f}

Final Loss: {result.get('metrics', {}).get('final_loss', current_loss_history[-1][1] if current_loss_history else 0):.2e}
"""
    ax.text(0.02, 0.95, metrics_text.strip(), fontsize=10, verticalalignment='top',
            fontfamily='monospace', transform=ax.transAxes)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_1d_standard(result, title, wavelength, pixel_size):
    """Standardized 1D visualization with complete analysis.

    Includes:
    - Row 1: Phase profile, Target vs Sim 1x (angles), Target vs Sim 2x (angles)
    - Row 2: Efficiency 1x, Efficiency 2x, Loss curve
    - Row 3: Parameters and metrics

    All order axes shown as actual diffraction angles (degrees).
    """
    phase = np.array(result['phase'])
    target = np.array(result['target_intensity'])
    simulated = np.array(result['simulated_intensity'])
    metrics = result.get('metrics', {})
    splitter_info = result.get('splitter_info', {})
    current_loss_history = get_loss_history()

    # Handle 2D arrays with 1 column
    if len(phase.shape) == 2:
        phase_1d = phase[:, 0] if phase.shape[1] == 1 else phase.mean(axis=1)
        target_1d = target[:, 0] if target.shape[1] == 1 else target.mean(axis=1)
        simulated_1d = simulated[:, 0] if simulated.shape[1] == 1 else simulated.mean(axis=1)
    else:
        phase_1d = phase
        target_1d = target
        simulated_1d = simulated

    h_phase = len(phase_1d)
    h_sim = len(target_1d)

    # Get period for angle conversion
    period_pixels = splitter_info.get('period_pixels', h_phase)
    period_m = period_pixels * pixel_size

    # Compute 2x upsampled with CROP to same angular range
    # This gives higher fidelity simulation at the same angle points
    sim_2x = evaluate_with_upsampling(phase, upsample_factor=2, crop_to_original=True)
    if len(sim_2x.shape) == 2:
        sim_2x_1d = sim_2x[:, sim_2x.shape[1]//2]
    else:
        sim_2x_1d = sim_2x
    # After cropping, h_2x = h_sim (same size)
    h_2x = len(sim_2x_1d)

    # Get order info
    order_positions = splitter_info.get('order_positions', [])
    order_efficiencies = metrics.get('order_efficiencies', [])
    working_orders = splitter_info.get('working_orders', [])

    # Compute 2x efficiency (same positions since we cropped)
    if order_positions:
        # With cropping, positions stay the same
        # For FFT k-space, use integration_radius=0 (single pixel) because orders
        # are at discrete k-space points and consecutive positions would overlap
        _, eff_2x, uni_2x = compute_efficiency_from_intensity(
            sim_2x_1d, [(p[0], 0) for p in order_positions], integration_radius=0
        )
    else:
        eff_2x = []
        uni_2x = 0.0

    # Compute angle axes
    phase_x_um = np.arange(h_phase) * pixel_size * 1e6

    # Convert all order positions to angles
    order_axis_1x = np.arange(h_sim) - h_sim // 2
    # With cropping, 2x output has same size and order mapping as 1x
    order_axis_2x = np.arange(h_2x) - h_2x // 2

    # Angle conversion for axes (same for both since cropped)
    angle_axis_1x = np.array([order_to_angle(m, wavelength, period_m) or 0
                              for m in order_axis_1x])
    angle_axis_1x = np.degrees(angle_axis_1x)

    # For cropped 2x, angle axis is the same as 1x
    angle_axis_2x = angle_axis_1x.copy()

    # Convert working orders to angles
    if working_orders:
        working_order_values = [o[0] for o in working_orders]
        working_angles = orders_to_angles_deg(working_order_values, wavelength, period_m)
    else:
        working_order_values = []
        working_angles = []

    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 0.6])

    # ===== Row 1: Phase, Target vs Sim 1x, Target vs Sim 2x =====

    # Phase profile
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(phase_x_um, phase_1d, 'b-', linewidth=1.5)
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('Phase (rad)')
    ax.set_title('Optimized Phase Profile')
    ax.set_ylim([0, 2*np.pi])
    ax.grid(True, alpha=0.3)

    # Target vs Simulated 1x (using angles)
    ax = fig.add_subplot(gs[0, 1])
    # Normalize for comparison
    target_norm = target_1d / (target_1d.max() + 1e-10)
    sim_norm = simulated_1d / (simulated_1d.max() + 1e-10)

    ax.plot(angle_axis_1x, target_norm, 'g-', linewidth=1, alpha=0.5, label='Target')
    ax.plot(angle_axis_1x, sim_norm, 'r-', linewidth=1, alpha=0.5, label='Simulated')

    # Scatter at order positions (using angles)
    if order_positions and working_angles:
        target_at_orders = [target_1d[p[0]] / (target_1d.max() + 1e-10)
                            for p in order_positions if 0 <= p[0] < h_sim]
        sim_at_orders = [simulated_1d[p[0]] / (simulated_1d.max() + 1e-10)
                         for p in order_positions if 0 <= p[0] < h_sim]

        ax.scatter(working_angles[:len(target_at_orders)], target_at_orders,
                   c='green', s=80, marker='o', label='Target@orders', zorder=5)
        ax.scatter(working_angles[:len(sim_at_orders)], sim_at_orders,
                   c='red', s=80, marker='x', label='Sim@orders', zorder=5)

    ax.set_xlabel('Angle θ (°)')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title('Target vs Simulated (1x)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Target vs Simulated 2x (using angles)
    # With cropping, 2x has same size as 1x
    ax = fig.add_subplot(gs[0, 2])
    # Since 2x is cropped to original size, target doesn't need interpolation
    target_2x_norm = target_1d / (target_1d.max() + 1e-10)
    sim_2x_norm = sim_2x_1d / (sim_2x_1d.max() + 1e-10)

    ax.plot(angle_axis_2x, target_2x_norm, 'g-', linewidth=1, alpha=0.5, label='Target')
    ax.plot(angle_axis_2x, sim_2x_norm, 'b-', linewidth=1, alpha=0.5, label='Simulated 2x')

    # Scatter at order positions (same as 1x since cropped)
    if working_angles and order_positions:
        sim_at_orders_2x = []
        for p in order_positions:
            idx = p[0]
            if 0 <= idx < h_2x:
                sim_at_orders_2x.append(sim_2x_1d[idx] / (sim_2x_1d.max() + 1e-10))
            else:
                sim_at_orders_2x.append(0)

        ax.scatter(working_angles[:len(sim_at_orders_2x)], sim_at_orders_2x,
                   c='blue', s=80, marker='x', label='Sim@orders 2x', zorder=5)

    ax.set_xlabel('Angle θ (°)')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title('Target vs Simulated (2x)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ===== Row 2: Efficiency 1x, Efficiency 2x, Loss =====

    # Efficiency 1x (bar chart uses order index, not angles)
    ax = fig.add_subplot(gs[1, 0])
    if order_efficiencies and working_orders:
        ax.bar(range(len(order_efficiencies)), order_efficiencies, color='steelblue', alpha=0.8, width=0.8)
        theoretical = 1.0 / len(order_efficiencies)
        ax.axhline(y=theoretical, color='green', linestyle=':', label=f'Theory: {theoretical:.4f}')
        ax.axhline(y=np.mean(order_efficiencies), color='red', linestyle='--',
                   label=f'Mean: {np.mean(order_efficiencies):.4f}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    total_eff = metrics.get('total_efficiency', 0)
    uni_1x = metrics.get('uniformity', 0)
    ax.set_title(f'Efficiency 1x (Total={total_eff:.3f}, Uni={uni_1x:.3f})')
    ax.set_xlabel('Order Index')
    ax.set_ylabel('Efficiency')

    # Efficiency 2x (bar chart uses order index)
    ax = fig.add_subplot(gs[1, 1])
    if eff_2x and working_orders:
        ax.bar(range(len(eff_2x)), eff_2x, color='forestgreen', alpha=0.8, width=0.8)
        theoretical = 1.0 / len(eff_2x)
        ax.axhline(y=theoretical, color='blue', linestyle=':', label=f'Theory: {theoretical:.4f}')
        ax.axhline(y=np.mean(eff_2x), color='red', linestyle='--',
                   label=f'Mean: {np.mean(eff_2x):.4f}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    total_eff_2x = sum(eff_2x) if eff_2x else 0
    ax.set_title(f'Efficiency 2x (Total={total_eff_2x:.3f}, Uni={uni_2x:.3f})')
    ax.set_xlabel('Order Index')
    ax.set_ylabel('Efficiency')

    # Loss curve
    ax = fig.add_subplot(gs[1, 2])
    if current_loss_history:
        iters = [x[0] for x in current_loss_history]
        losses = [x[1] for x in current_loss_history]
        ax.semilogy(iters, losses, 'b-', linewidth=1.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
    ax.set_title('Loss Curve')

    # ===== Row 3: Parameters =====

    ax = fig.add_subplot(gs[2, :2])
    ax.axis('off')
    param_text = format_structured_params(result)

    # Add angle info
    if working_angles:
        angle_range = f"[{min(working_angles):.2f}°, {max(working_angles):.2f}°]"
    else:
        angle_range = "N/A"

    summary_text = f"""
STRUCTURED PARAMETERS:
{param_text}

PHYSICAL PARAMETERS:
- Wavelength: {wavelength*1e9:.1f} nm
- Pixel Size: {pixel_size*1e6:.2f} μm
- DOE Size: {h_phase * pixel_size * 1e6:.0f} μm
- Period: {period_m*1e6:.2f} μm ({period_pixels} px)
- Working Angles: {angle_range}
"""
    ax.text(0.02, 0.95, summary_text.strip(), fontsize=9, verticalalignment='top',
            fontfamily='monospace', transform=ax.transAxes)

    ax = fig.add_subplot(gs[2, 2])
    ax.axis('off')
    metrics_text = f"""
METRICS COMPARISON:

               1x         2x
Total Eff:     {total_eff:.4f}     {total_eff_2x:.4f}
Uniformity:    {uni_1x:.4f}     {uni_2x:.4f}
"""
    ax.text(0.02, 0.95, metrics_text.strip(), fontsize=10, verticalalignment='top',
            fontfamily='monospace', transform=ax.transAxes)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_strategy2_standard(result, title, wavelength, pixel_size,
                                  working_distance, target_span):
    """Standardized Strategy 2 visualization.

    Shows:
    - Periodic phase and FFT evaluation
    - Full device with Fresnel
    - Both 1x and 2x analysis
    """
    phase = np.array(result['phase'])
    simulated = np.array(result['simulated_intensity'])
    metrics = result.get('metrics', {})
    splitter_info = result.get('splitter_info', {})
    computed = result.get('computed_params', {})
    current_loss_history = get_loss_history()

    h_phase, w_phase = phase.shape
    h_sim, w_sim = simulated.shape

    # Compute 2x upsampled with crop to same angular range
    sim_2x = evaluate_with_upsampling(phase, upsample_factor=2, crop_to_original=True)
    h_2x, w_2x = sim_2x.shape  # Same as original since cropped

    # Get order info
    order_positions = splitter_info.get('order_positions', [])
    order_efficiencies = metrics.get('order_efficiencies', [])
    working_orders = splitter_info.get('working_orders', [])

    # Compute 2x efficiency
    # For FFT k-space, use integration_radius=0 to avoid overlap at consecutive orders
    # Since we cropped to original size, positions stay the same
    if order_positions:
        _, eff_2x, uni_2x = compute_efficiency_from_intensity(
            sim_2x, order_positions, integration_radius=0
        )
    else:
        eff_2x = []
        uni_2x = 0.0

    # Extents (same for 1x and 2x since cropped)
    phase_extent = [0, w_phase * pixel_size * 1e6, 0, h_phase * pixel_size * 1e6]
    order_extent = [-w_sim//2, w_sim//2, -h_sim//2, h_sim//2]
    order_extent_2x = order_extent  # Same since cropped

    # Full device
    num_periods = computed.get('num_periods', (4, 4))
    period_pixels = splitter_info.get('period_pixels', h_phase)
    nx, ny = num_periods if isinstance(num_periods, tuple) else (num_periods, num_periods)
    nx, ny = max(1, min(nx, 6)), max(1, min(ny, 6))
    full_phase = np.tile(phase, (ny, nx))

    # Add Fresnel
    h_full, w_full = full_phase.shape
    y = np.linspace(-h_full/2, h_full/2, h_full) * pixel_size
    x = np.linspace(-w_full/2, w_full/2, w_full) * pixel_size
    xx, yy = np.meshgrid(x, y)
    fresnel_phase = -np.pi * (xx**2 + yy**2) / (wavelength * working_distance)
    full_phase_with_fresnel = (full_phase + fresnel_phase) % (2 * np.pi)
    full_extent = [0, w_full * pixel_size * 1e6, 0, h_full * pixel_size * 1e6]

    # Create figure
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 4, figure=fig, height_ratios=[1, 1, 1, 0.6])

    # ===== Row 1: Periodic phase, Simulated 1x, Simulated 2x, Loss =====

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(phase, cmap='twilight', vmin=0, vmax=2*np.pi,
                   extent=phase_extent, origin='lower')
    ax.set_title('Periodic Phase (1 period)')
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(simulated, cmap='hot', extent=order_extent, origin='lower')
    ax.set_title('FFT Simulated 1x (linear)')
    ax.set_xlabel('Order nx')
    ax.set_ylabel('Order ny')
    plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(sim_2x, cmap='hot', extent=order_extent_2x, origin='lower')
    ax.set_title('FFT Simulated 2x (linear)')
    ax.set_xlabel('Order nx')
    ax.set_ylabel('Order ny')
    ax.set_xlim(order_extent[0]*2, order_extent[1]*2)
    ax.set_ylim(order_extent[2]*2, order_extent[3]*2)
    plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(gs[0, 3])
    if current_loss_history:
        iters = [x[0] for x in current_loss_history]
        losses = [x[1] for x in current_loss_history]
        ax.semilogy(iters, losses, 'b-', linewidth=1.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
    ax.set_title('Loss Curve')

    # ===== Row 2: Full device, With Fresnel, Efficiency 1x, Efficiency 2x =====

    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(full_phase % (2*np.pi), cmap='twilight', vmin=0, vmax=2*np.pi,
                   extent=full_extent, origin='lower')
    ax.set_title(f'Full Device ({nx}x{ny} periods)')
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(full_phase_with_fresnel, cmap='twilight', vmin=0, vmax=2*np.pi,
                   extent=full_extent, origin='lower')
    ax.set_title('With Fresnel Lens')
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(gs[1, 2])
    if order_efficiencies:
        n = len(order_efficiencies)
        ax.bar(range(n), order_efficiencies, color='steelblue', alpha=0.8)
        theoretical = 1.0 / n
        ax.axhline(y=theoretical, color='green', linestyle=':', label=f'Theory: {theoretical:.4f}')
        ax.axhline(y=np.mean(order_efficiencies), color='red', linestyle='--',
                   label=f'Mean: {np.mean(order_efficiencies):.4f}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    total_eff = metrics.get('total_efficiency', 0)
    uni_1x = metrics.get('uniformity', 0)
    ax.set_title(f'Efficiency 1x (Total={total_eff:.3f})')
    ax.set_xlabel('Order Index')
    ax.set_ylabel('Efficiency')

    ax = fig.add_subplot(gs[1, 3])
    if eff_2x:
        n = len(eff_2x)
        ax.bar(range(n), eff_2x, color='forestgreen', alpha=0.8)
        theoretical = 1.0 / n
        ax.axhline(y=theoretical, color='blue', linestyle=':', label=f'Theory: {theoretical:.4f}')
        ax.axhline(y=np.mean(eff_2x), color='red', linestyle='--',
                   label=f'Mean: {np.mean(eff_2x):.4f}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    total_eff_2x = sum(eff_2x) if eff_2x else 0
    ax.set_title(f'Efficiency 2x (Total={total_eff_2x:.3f})')
    ax.set_xlabel('Order Index')
    ax.set_ylabel('Efficiency')

    # ===== Row 3: Order positions 1x, 2x, SFR simulation, placeholder =====

    ax = fig.add_subplot(gs[2, 0])
    if order_positions and order_efficiencies:
        pos_y = [p[0] - h_sim//2 for p in order_positions]
        pos_x = [p[1] - w_sim//2 for p in order_positions]
        sc = ax.scatter(pos_x, pos_y, c=order_efficiencies, cmap='viridis',
                       s=100, edgecolors='black')
        plt.colorbar(sc, ax=ax, label='Efficiency')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    ax.set_title('Order Positions 1x')
    ax.set_xlabel('Order nx')
    ax.set_ylabel('Order ny')

    ax = fig.add_subplot(gs[2, 1])
    if order_positions and eff_2x:
        pos_y = [p[0] - h_2x//2 for p in order_positions]
        pos_x = [p[1] - w_2x//2 for p in order_positions]
        sc = ax.scatter(pos_x, pos_y, c=eff_2x, cmap='viridis',
                       s=100, edgecolors='black')
        plt.colorbar(sc, ax=ax, label='Efficiency')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    ax.set_title('Order Positions 2x')
    ax.set_xlabel('Order nx')
    ax.set_ylabel('Order ny')

    # SFR simulation of full device
    ax = fig.add_subplot(gs[2, 2])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    full_phase_t = torch.tensor(full_phase_with_fresnel, device=device, dtype=torch.float32)
    full_phase_t = full_phase_t.unsqueeze(0).unsqueeze(0)
    field = torch.exp(1j * full_phase_t.to(torch.complex64))
    output_field = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field)))
    sfr_intensity = (output_field.abs() ** 2).squeeze().cpu().numpy()
    im = ax.imshow(sfr_intensity, cmap='hot', origin='lower')
    ax.set_title('Full Device FFT (physical space)')
    ax.set_xlabel('Index')
    ax.set_ylabel('Index')
    plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(gs[2, 3])
    ax.axis('off')

    # ===== Row 4: Parameters =====

    ax = fig.add_subplot(gs[3, :2])
    ax.axis('off')
    param_text = format_structured_params(result)
    summary_text = f"""
STRATEGY 2: Periodic + Fresnel

STRUCTURED PARAMETERS:
{param_text}

PHYSICAL PARAMETERS:
- Wavelength: {wavelength*1e9:.1f} nm
- Pixel Size: {pixel_size*1e6:.2f} μm
- Working Distance: {working_distance*1e3:.2f} mm
- Target Span: {target_span*1e3:.2f} mm
- Period: {period_pixels} px
- Num Periods: {nx} x {ny}
"""
    ax.text(0.02, 0.95, summary_text.strip(), fontsize=9, verticalalignment='top',
            fontfamily='monospace', transform=ax.transAxes)

    ax = fig.add_subplot(gs[3, 2:])
    ax.axis('off')
    metrics_text = f"""
METRICS COMPARISON:

               1x         2x
Total Eff:     {total_eff:.4f}     {total_eff_2x:.4f}
Uniformity:    {uni_1x:.4f}     {uni_2x:.4f}
"""
    ax.text(0.02, 0.95, metrics_text.strip(), fontsize=10, verticalalignment='top',
            fontfamily='monospace', transform=ax.transAxes)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


# =============================================================================
# Test Cases
# =============================================================================

def test_fft_2d_natural():
    """Type A: FFT 2D splitter with natural grid."""
    print("\n" + "="*70)
    print("TEST: FFT 2D Natural Grid (5x5)")
    print("="*70)

    wavelength = 532e-9
    pixel_size = 1e-6

    user_input = {
        'doe_type': 'splitter_2d',
        'wavelength': wavelength,
        'device_diameter': 256e-6,
        'pixel_size': pixel_size,
        'target_spec': {
            'num_spots': [5, 5],
            'target_type': 'angle',
            'target_span': [0.2, 0.2],
            'grid_mode': 'natural'
        },
        'optimization': {'phase_iters': 500}
    }

    response = run_optimization(user_input, progress_callback=print_progress)

    if response.success:
        result = response.result.to_dict()
        print(f"\nResults: Eff={result['metrics']['total_efficiency']:.4f}, "
              f"Uniformity={result['metrics']['uniformity']:.4f}")
        return visualize_2d_standard(result, "FFT 2D: Natural Grid (5x5)",
                                      wavelength, pixel_size), result
    else:
        print(f"ERROR: {[e.message for e in response.errors]}")
        return None, None


def test_fft_2d_uniform_tol5():
    """Type A: FFT 2D with uniform grid, tolerance=5%."""
    print("\n" + "="*70)
    print("TEST: FFT 2D Uniform Grid (tolerance=5%)")
    print("="*70)

    wavelength = 532e-9
    pixel_size = 1e-6

    user_input = {
        'doe_type': 'splitter_2d',
        'wavelength': wavelength,
        'device_diameter': 256e-6,
        'pixel_size': pixel_size,
        'target_spec': {
            'num_spots': [5, 5],
            'target_type': 'angle',
            'target_span': [0.15, 0.15],
            'tolerance': 0.05,
            'grid_mode': 'uniform'
        },
        'optimization': {'phase_iters': 500}
    }

    response = run_optimization(user_input, progress_callback=print_progress)

    if response.success:
        result = response.result.to_dict()
        period = result.get('splitter_info', {}).get('period_pixels', 'N/A')
        print(f"\nPeriod: {period}px, Eff={result['metrics']['total_efficiency']:.4f}")
        return visualize_2d_standard(result, f"FFT 2D: Uniform Grid (tol=5%, period={period}px)",
                                      wavelength, pixel_size), result
    else:
        print(f"ERROR: {[e.message for e in response.errors]}")
        return None, None


def test_fft_2d_uniform_tol1():
    """Type A: FFT 2D with uniform grid, tolerance=1%."""
    print("\n" + "="*70)
    print("TEST: FFT 2D Uniform Grid (tolerance=1%)")
    print("="*70)

    wavelength = 532e-9
    pixel_size = 1e-6

    user_input = {
        'doe_type': 'splitter_2d',
        'wavelength': wavelength,
        'device_diameter': 256e-6,
        'pixel_size': pixel_size,
        'target_spec': {
            'num_spots': [5, 5],
            'target_type': 'angle',
            'target_span': [0.15, 0.15],
            'tolerance': 0.01,
            'grid_mode': 'uniform'
        },
        'optimization': {'phase_iters': 500}
    }

    response = run_optimization(user_input, progress_callback=print_progress)

    if response.success:
        result = response.result.to_dict()
        period = result.get('splitter_info', {}).get('period_pixels', 'N/A')
        print(f"\nPeriod: {period}px, Eff={result['metrics']['total_efficiency']:.4f}")
        return visualize_2d_standard(result, f"FFT 2D: Uniform Grid (tol=1%, period={period}px)",
                                      wavelength, pixel_size), result
    else:
        print(f"ERROR: {[e.message for e in response.errors]}")
        return None, None


def test_fft_2d_bigpixel_2um():
    """Type A: FFT 2D with big pixel (2um)."""
    print("\n" + "="*70)
    print("TEST: FFT 2D Big Pixel (2μm)")
    print("="*70)

    wavelength = 532e-9
    pixel_size = 2e-6

    user_input = {
        'doe_type': 'splitter_2d',
        'wavelength': wavelength,
        'device_diameter': 256e-6,
        'pixel_size': pixel_size,
        'target_spec': {
            'num_spots': [5, 5],
            'target_type': 'angle',
            'target_span': [0.1, 0.1],
            'grid_mode': 'natural'
        },
        'optimization': {'phase_iters': 500}
    }

    response = run_optimization(user_input, progress_callback=print_progress)

    if response.success:
        result = response.result.to_dict()
        period = result.get('splitter_info', {}).get('period_pixels', 'N/A')
        print(f"\nPeriod: {period}px, Eff={result['metrics']['total_efficiency']:.4f}")
        return visualize_2d_standard(result, f"FFT 2D: Big Pixel (2μm, period={period}px)",
                                      wavelength, pixel_size), result
    else:
        print(f"ERROR: {[e.message for e in response.errors]}")
        return None, None


def test_fft_2d_bigpixel_4um():
    """Type A: FFT 2D with big pixel (4um)."""
    print("\n" + "="*70)
    print("TEST: FFT 2D Big Pixel (4μm)")
    print("="*70)

    wavelength = 532e-9
    pixel_size = 4e-6

    user_input = {
        'doe_type': 'splitter_2d',
        'wavelength': wavelength,
        'device_diameter': 256e-6,
        'pixel_size': pixel_size,
        'target_spec': {
            'num_spots': [5, 5],
            'target_type': 'angle',
            'target_span': [0.1, 0.1],
            'grid_mode': 'natural'
        },
        'optimization': {'phase_iters': 500}
    }

    response = run_optimization(user_input, progress_callback=print_progress)

    if response.success:
        result = response.result.to_dict()
        period = result.get('splitter_info', {}).get('period_pixels', 'N/A')
        print(f"\nPeriod: {period}px, Eff={result['metrics']['total_efficiency']:.4f}")
        return visualize_2d_standard(result, f"FFT 2D: Big Pixel (4μm, period={period}px)",
                                      wavelength, pixel_size), result
    else:
        print(f"ERROR: {[e.message for e in response.errors]}")
        return None, None


def test_fft_1d_natural():
    """Type A: FFT 1D splitter with natural grid."""
    print("\n" + "="*70)
    print("TEST: FFT 1D Natural Grid (7 spots)")
    print("="*70)

    wavelength = 532e-9
    pixel_size = 1e-6

    user_input = {
        'doe_type': 'splitter_1d',
        'wavelength': wavelength,
        'device_diameter': 256e-6,
        'pixel_size': pixel_size,
        'target_spec': {
            'num_spots': 7,
            'target_type': 'angle',
            'target_span': 0.15,
            'grid_mode': 'natural'
        },
        'optimization': {'phase_iters': 500}
    }

    response = run_optimization(user_input, progress_callback=print_progress)

    if response.success:
        result = response.result.to_dict()
        print(f"\nResults: Eff={result['metrics']['total_efficiency']:.4f}, "
              f"Uniformity={result['metrics']['uniformity']:.4f}")
        return visualize_1d_standard(result, "FFT 1D: Natural Grid (7 spots)",
                                      wavelength, pixel_size), result
    else:
        print(f"ERROR: {[e.message for e in response.errors]}")
        return None, None


def visualize_1d_with_3x(result, title, wavelength, pixel_size):
    """Visualize 1D with 2x optimization and 3x analysis.

    This visualization shows:
    - Phase profile
    - Target vs Simulated 2x (optimization resolution) with angles
    - Target vs Simulated 3x (analysis resolution) with angles
    - Efficiency comparison at both resolutions
    """
    phase = np.array(result['phase'])
    target = np.array(result['target_intensity'])
    simulated = np.array(result['simulated_intensity'])
    metrics = result.get('metrics', {})
    splitter_info = result.get('splitter_info', {})
    current_loss_history = get_loss_history()

    # Handle 2D arrays
    if len(phase.shape) == 2:
        phase_1d = phase[:, 0] if phase.shape[1] == 1 else phase.mean(axis=1)
        target_1d = target[:, 0] if target.shape[1] == 1 else target.mean(axis=1)
        simulated_1d = simulated[:, 0] if simulated.shape[1] == 1 else simulated.mean(axis=1)
    else:
        phase_1d = phase
        target_1d = target
        simulated_1d = simulated

    h_phase = len(phase_1d)
    h_sim = len(target_1d)

    # Get period for angle conversion
    period_pixels = splitter_info.get('period_pixels', h_phase)
    period_m = period_pixels * pixel_size

    # Compute 2x and 3x upsampled with CROP to same angular range
    sim_2x = evaluate_with_upsampling(phase, upsample_factor=2, crop_to_original=True)
    sim_3x = evaluate_with_upsampling(phase, upsample_factor=3, crop_to_original=True)

    if len(sim_2x.shape) == 2:
        sim_2x_1d = sim_2x[:, sim_2x.shape[1]//2]
    else:
        sim_2x_1d = sim_2x

    if len(sim_3x.shape) == 2:
        sim_3x_1d = sim_3x[:, sim_3x.shape[1]//2]
    else:
        sim_3x_1d = sim_3x

    # After cropping, sizes are the same as original
    h_2x = len(sim_2x_1d)
    h_3x = len(sim_3x_1d)

    # Get order info
    order_positions = splitter_info.get('order_positions', [])
    order_efficiencies = metrics.get('order_efficiencies', [])
    working_orders = splitter_info.get('working_orders', [])

    # Compute efficiencies at 2x and 3x (same positions since cropped)
    # For FFT k-space, use integration_radius=0 to avoid overlap at consecutive orders
    if order_positions:
        _, eff_2x, uni_2x = compute_efficiency_from_intensity(
            sim_2x_1d, [(p[0], 0) for p in order_positions], integration_radius=0
        )
        _, eff_3x, uni_3x = compute_efficiency_from_intensity(
            sim_3x_1d, [(p[0], 0) for p in order_positions], integration_radius=0
        )
    else:
        eff_2x, eff_3x = [], []
        uni_2x, uni_3x = 0.0, 0.0

    # Axes - convert to angles (same for all since cropped)
    phase_x_um = np.arange(h_phase) * pixel_size * 1e6
    order_axis = np.arange(h_sim) - h_sim // 2

    # Angle conversion for axis (same for all resolutions)
    angle_axis = np.array([order_to_angle(m, wavelength, period_m) or 0
                           for m in order_axis])
    angle_axis = np.degrees(angle_axis)

    # Convert working orders to angles
    if working_orders:
        working_order_values = [o[0] for o in working_orders]
        working_angles = orders_to_angles_deg(working_order_values, wavelength, period_m)
    else:
        working_order_values = []
        working_angles = []

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 0.6])

    # ===== Row 1: Phase, Target vs Sim 2x, Target vs Sim 3x, Loss =====

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(phase_x_um, phase_1d, 'b-', linewidth=1.5)
    ax.set_xlabel('Position (μm)')
    ax.set_ylabel('Phase (rad)')
    ax.set_title('Optimized Phase Profile')
    ax.set_ylim([0, 2*np.pi])
    ax.grid(True, alpha=0.3)

    # Target vs Simulated 2x (using angles) - cropped to same angular range
    ax = fig.add_subplot(gs[0, 1])
    target_norm = target_1d / (target_1d.max() + 1e-10)
    sim_2x_norm = sim_2x_1d / (sim_2x_1d.max() + 1e-10)

    ax.plot(angle_axis, target_norm, 'g-', linewidth=1, alpha=0.5, label='Target')
    ax.plot(angle_axis, sim_2x_norm, 'r-', linewidth=1, alpha=0.5, label='Simulated 2x')

    if working_angles and order_positions:
        sim_at_2x = []
        for p in order_positions:
            idx = p[0]
            if 0 <= idx < h_2x:
                sim_at_2x.append(sim_2x_norm[idx])
        ax.scatter(working_angles[:len(sim_at_2x)], sim_at_2x,
                   c='red', s=80, marker='x', label='Sim@orders', zorder=5)

    ax.set_xlabel('Angle θ (°)')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title('Target vs Simulated (2x - Opt Resolution)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Target vs Simulated 3x (using angles) - cropped to same angular range
    ax = fig.add_subplot(gs[0, 2])
    sim_3x_norm = sim_3x_1d / (sim_3x_1d.max() + 1e-10)

    ax.plot(angle_axis, target_norm, 'g-', linewidth=1, alpha=0.5, label='Target')
    ax.plot(angle_axis, sim_3x_norm, 'b-', linewidth=1, alpha=0.5, label='Simulated 3x')

    if working_angles and order_positions:
        sim_at_3x = []
        for p in order_positions:
            idx = p[0]
            if 0 <= idx < h_3x:
                sim_at_3x.append(sim_3x_norm[idx])
        ax.scatter(working_angles[:len(sim_at_3x)], sim_at_3x,
                   c='blue', s=80, marker='x', label='Sim@orders', zorder=5)

    ax.set_xlabel('Angle θ (°)')
    ax.set_ylabel('Normalized Intensity')
    ax.set_title('Target vs Simulated (3x - Analysis)')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Loss curve
    ax = fig.add_subplot(gs[0, 3])
    if current_loss_history:
        iters = [x[0] for x in current_loss_history]
        losses = [x[1] for x in current_loss_history]
        ax.semilogy(iters, losses, 'b-', linewidth=1.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
    ax.set_title('Loss Curve')

    # ===== Row 2: Efficiency 2x, Efficiency 3x (bar charts use index) =====

    ax = fig.add_subplot(gs[1, 0])
    if eff_2x:
        ax.bar(range(len(eff_2x)), eff_2x, color='steelblue', alpha=0.8, width=0.8)
        theoretical = 1.0 / len(eff_2x)
        ax.axhline(y=theoretical, color='green', linestyle=':', label=f'Theory: {theoretical:.4f}')
        ax.axhline(y=np.mean(eff_2x), color='red', linestyle='--',
                   label=f'Mean: {np.mean(eff_2x):.4f}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    total_eff_2x = sum(eff_2x) if eff_2x else 0
    ax.set_title(f'Efficiency 2x (Total={total_eff_2x:.3f}, Uni={uni_2x:.3f})')
    ax.set_xlabel('Order Index')
    ax.set_ylabel('Efficiency')

    ax = fig.add_subplot(gs[1, 1])
    if eff_3x:
        ax.bar(range(len(eff_3x)), eff_3x, color='forestgreen', alpha=0.8, width=0.8)
        theoretical = 1.0 / len(eff_3x)
        ax.axhline(y=theoretical, color='blue', linestyle=':', label=f'Theory: {theoretical:.4f}')
        ax.axhline(y=np.mean(eff_3x), color='red', linestyle='--',
                   label=f'Mean: {np.mean(eff_3x):.4f}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    total_eff_3x = sum(eff_3x) if eff_3x else 0
    ax.set_title(f'Efficiency 3x (Total={total_eff_3x:.3f}, Uni={uni_3x:.3f})')
    ax.set_xlabel('Order Index')
    ax.set_ylabel('Efficiency')

    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')

    ax = fig.add_subplot(gs[1, 3])
    ax.axis('off')

    # ===== Row 3: Parameters =====

    ax = fig.add_subplot(gs[2, :2])
    ax.axis('off')
    param_text = format_structured_params(result)

    # Add angle info
    if working_angles:
        angle_range = f"[{min(working_angles):.2f}°, {max(working_angles):.2f}°]"
    else:
        angle_range = "N/A"

    summary_text = f"""
UPSAMPLING COMPARISON (2x vs 3x Analysis)

STRUCTURED PARAMETERS:
{param_text}

PHYSICAL PARAMETERS:
- Wavelength: {wavelength*1e9:.1f} nm
- Pixel Size: {pixel_size*1e6:.2f} μm
- DOE Size: {h_phase * pixel_size * 1e6:.0f} μm
- Period: {period_m*1e6:.2f} μm ({period_pixels} px)
- Working Angles: {angle_range}

NOTE: This test uses 2x upsampling during optimization,
then 3x upsampling for higher-resolution analysis.
"""
    ax.text(0.02, 0.95, summary_text.strip(), fontsize=9, verticalalignment='top',
            fontfamily='monospace', transform=ax.transAxes)

    ax = fig.add_subplot(gs[2, 2:])
    ax.axis('off')
    total_eff_1x = metrics.get('total_efficiency', 0)
    uni_1x = metrics.get('uniformity', 0)
    metrics_text = f"""
METRICS COMPARISON:

               1x         2x         3x
Total Eff:     {total_eff_1x:.4f}     {total_eff_2x:.4f}     {total_eff_3x:.4f}
Uniformity:    {uni_1x:.4f}     {uni_2x:.4f}     {uni_3x:.4f}
"""
    ax.text(0.02, 0.95, metrics_text.strip(), fontsize=10, verticalalignment='top',
            fontfamily='monospace', transform=ax.transAxes)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def visualize_2d_with_3x(result, title, wavelength, pixel_size):
    """Visualize 2D with 2x optimization and 3x analysis.

    This visualization shows:
    - Phase heatmap
    - Simulated 2x (optimization resolution)
    - Simulated 3x (analysis resolution)
    - Efficiency comparison at both resolutions
    """
    phase = np.array(result['phase'])
    target = np.array(result['target_intensity'])
    simulated = np.array(result['simulated_intensity'])
    metrics = result.get('metrics', {})
    splitter_info = result.get('splitter_info', {})
    current_loss_history = get_loss_history()

    h_phase, w_phase = phase.shape
    h_sim, w_sim = simulated.shape

    # Get period for angle conversion
    period_pixels = splitter_info.get('period_pixels', h_phase)
    period_m = period_pixels * pixel_size

    # Compute 2x and 3x upsampled with CROP to same angular range
    sim_2x = evaluate_with_upsampling(phase, upsample_factor=2, crop_to_original=True)
    sim_3x = evaluate_with_upsampling(phase, upsample_factor=3, crop_to_original=True)

    # Get order info
    order_positions = splitter_info.get('order_positions', [])
    order_efficiencies = metrics.get('order_efficiencies', [])

    # Compute efficiencies at 2x and 3x (same positions since cropped)
    # For FFT k-space, use integration_radius=0 to avoid overlap at consecutive orders
    if order_positions:
        _, eff_2x, uni_2x = compute_efficiency_from_intensity(
            sim_2x, order_positions, integration_radius=0
        )
        _, eff_3x, uni_3x = compute_efficiency_from_intensity(
            sim_3x, order_positions, integration_radius=0
        )
    else:
        eff_2x, eff_3x = [], []
        uni_2x, uni_3x = 0.0, 0.0

    # Create figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 0.6])

    # ===== Row 1: Phase, Simulated 2x, Simulated 3x, Loss =====

    # Phase heatmap
    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(phase, cmap='twilight', vmin=0, vmax=2*np.pi)
    ax.set_title('Optimized Phase')
    ax.set_xlabel('Pixel X')
    ax.set_ylabel('Pixel Y')
    plt.colorbar(im, ax=ax, label='Phase (rad)')

    # Simulated 2x (optimization resolution)
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(sim_2x, cmap='hot')
    ax.set_title('Simulated 2x (Opt Resolution)')
    ax.set_xlabel('Pixel X')
    ax.set_ylabel('Pixel Y')
    plt.colorbar(im, ax=ax, label='Intensity')

    # Simulated 3x (analysis resolution)
    ax = fig.add_subplot(gs[0, 2])
    im = ax.imshow(sim_3x, cmap='hot')
    ax.set_title('Simulated 3x (Analysis)')
    ax.set_xlabel('Pixel X')
    ax.set_ylabel('Pixel Y')
    plt.colorbar(im, ax=ax, label='Intensity')

    # Loss curve
    ax = fig.add_subplot(gs[0, 3])
    if current_loss_history:
        iters = [x[0] for x in current_loss_history]
        losses = [x[1] for x in current_loss_history]
        ax.semilogy(iters, losses, 'b-', linewidth=1.5)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Loss')
        ax.grid(True, alpha=0.3)
    ax.set_title('Loss Curve')

    # ===== Row 2: Efficiency 2x, Efficiency 3x (bar charts) =====

    ax = fig.add_subplot(gs[1, 0])
    if eff_2x:
        ax.bar(range(len(eff_2x)), eff_2x, color='steelblue', alpha=0.8, width=0.8)
        theoretical = 1.0 / len(eff_2x)
        ax.axhline(y=theoretical, color='green', linestyle=':', label=f'Theory: {theoretical:.4f}')
        ax.axhline(y=np.mean(eff_2x), color='red', linestyle='--',
                   label=f'Mean: {np.mean(eff_2x):.4f}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    total_eff_2x = sum(eff_2x) if eff_2x else 0
    ax.set_title(f'Efficiency 2x (Total={total_eff_2x:.3f}, Uni={uni_2x:.3f})')
    ax.set_xlabel('Order Index')
    ax.set_ylabel('Efficiency')

    ax = fig.add_subplot(gs[1, 1])
    if eff_3x:
        ax.bar(range(len(eff_3x)), eff_3x, color='forestgreen', alpha=0.8, width=0.8)
        theoretical = 1.0 / len(eff_3x)
        ax.axhline(y=theoretical, color='blue', linestyle=':', label=f'Theory: {theoretical:.4f}')
        ax.axhline(y=np.mean(eff_3x), color='red', linestyle='--',
                   label=f'Mean: {np.mean(eff_3x):.4f}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    total_eff_3x = sum(eff_3x) if eff_3x else 0
    ax.set_title(f'Efficiency 3x (Total={total_eff_3x:.3f}, Uni={uni_3x:.3f})')
    ax.set_xlabel('Order Index')
    ax.set_ylabel('Efficiency')

    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')

    ax = fig.add_subplot(gs[1, 3])
    ax.axis('off')

    # ===== Row 3: Parameters =====

    ax = fig.add_subplot(gs[2, :2])
    ax.axis('off')
    param_text = format_structured_params(result)

    summary_text = f"""
UPSAMPLING COMPARISON (2x vs 3x Analysis)

STRUCTURED PARAMETERS:
{param_text}

PHYSICAL PARAMETERS:
- Wavelength: {wavelength*1e9:.1f} nm
- Pixel Size: {pixel_size*1e6:.2f} um
- DOE Size: {h_phase * pixel_size * 1e6:.0f} x {w_phase * pixel_size * 1e6:.0f} um
- Period: {period_m*1e6:.2f} um ({period_pixels} px)

NOTE: This test uses 2x upsampling during optimization,
then 3x upsampling for higher-resolution analysis.
"""
    ax.text(0.02, 0.95, summary_text.strip(), fontsize=9, verticalalignment='top',
            fontfamily='monospace', transform=ax.transAxes)

    ax = fig.add_subplot(gs[2, 2:])
    ax.axis('off')
    total_eff_1x = metrics.get('total_efficiency', 0)
    uni_1x = metrics.get('uniformity', 0)
    metrics_text = f"""
METRICS COMPARISON:

               1x         2x         3x
Total Eff:     {total_eff_1x:.4f}     {total_eff_2x:.4f}     {total_eff_3x:.4f}
Uniformity:    {uni_1x:.4f}     {uni_2x:.4f}     {uni_3x:.4f}
"""
    ax.text(0.02, 0.95, metrics_text.strip(), fontsize=10, verticalalignment='top',
            fontfamily='monospace', transform=ax.transAxes)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def test_fft_2d_upsample_2x():
    """FFT 2D with 2x upsampling during optimization (via smaller pixel) + 3x analysis.

    This simulates 2x upsampling by using 0.5um pixel size instead of 1um,
    effectively doubling the resolution during optimization.
    Analysis is done at 3x of the optimization resolution.
    """
    print("\n" + "="*70)
    print("TEST: FFT 2D with 2x Upsampling (5x5) + 3x Analysis")
    print("="*70)

    wavelength = 532e-9
    # Use 0.5um pixel size to simulate 2x upsampling (compared to 1um base)
    pixel_size = 0.5e-6  # 2x finer than typical 1um

    user_input = {
        'doe_type': 'splitter_2d',
        'wavelength': wavelength,
        'device_diameter': 128e-6,  # Smaller device for faster computation
        'pixel_size': pixel_size,  # This gives 256x256 instead of 128x128
        'target_spec': {
            'num_spots': [5, 5],
            'target_type': 'angle',
            'target_span': [0.1, 0.1],  # Moderate angle span
            'grid_mode': 'natural'
        },
        'optimization': {'phase_iters': 500}
    }

    response = run_optimization(user_input, progress_callback=print_progress)

    if response.success:
        result = response.result.to_dict()
        print(f"\nResults: Eff={result['metrics']['total_efficiency']:.4f}, "
              f"Uniformity={result['metrics']['uniformity']:.4f}")
        period = result.get('splitter_info', {}).get('period_pixels', 'N/A')
        print(f"Period: {period} pixels")
        return visualize_2d_with_3x(result, "FFT 2D: 2x Upsampling (0.5um) + 3x Analysis",
                                     wavelength, pixel_size), result
    else:
        print(f"ERROR: {[e.message for e in response.errors]}")
        return None, None


def test_fft_1d_upsample_2x():
    """FFT 1D with 2x upsampling during optimization (via smaller pixel) + 3x analysis.

    This simulates 2x upsampling by using 0.5μm pixel size instead of 1μm,
    effectively doubling the resolution during optimization.
    Analysis is done at 3x of the optimization resolution.
    """
    print("\n" + "="*70)
    print("TEST: FFT 1D with 2x Upsampling (7 spots) + 3x Analysis")
    print("="*70)

    wavelength = 532e-9
    # Use 0.5μm pixel size to simulate 2x upsampling (compared to 1μm base)
    pixel_size = 0.5e-6  # 2x finer than typical 1μm

    user_input = {
        'doe_type': 'splitter_1d',
        'wavelength': wavelength,
        'device_diameter': 256e-6,
        'pixel_size': pixel_size,  # This gives 512 pixels instead of 256
        'target_spec': {
            'num_spots': 7,
            'target_type': 'angle',
            'target_span': 0.15,
            'grid_mode': 'natural'
        },
        'optimization': {'phase_iters': 500}
    }

    response = run_optimization(user_input, progress_callback=print_progress)

    if response.success:
        result = response.result.to_dict()
        print(f"\nResults: Eff={result['metrics']['total_efficiency']:.4f}, "
              f"Uniformity={result['metrics']['uniformity']:.4f}")
        return visualize_1d_with_3x(result, "FFT 1D: 2x Upsampling (0.5μm) + 3x Analysis",
                                     wavelength, pixel_size), result
    else:
        print(f"ERROR: {[e.message for e in response.errors]}")
        return None, None


def test_fft_1d_small_angle():
    """FFT 1D with smaller diffraction angle to test DOF issue.

    Using smaller angle (0.05 rad ≈ 2.9°) compared to 0.15 rad (≈ 8.6°)
    to test if the efficiency problem is related to large angle requirements.
    """
    print("\n" + "="*70)
    print("TEST: FFT 1D with Small Angle (7 spots, 0.05 rad)")
    print("="*70)

    wavelength = 532e-9
    pixel_size = 0.5e-6  # 2x finer (512 pixels)

    user_input = {
        'doe_type': 'splitter_1d',
        'wavelength': wavelength,
        'device_diameter': 256e-6,
        'pixel_size': pixel_size,
        'target_spec': {
            'num_spots': 7,
            'target_type': 'angle',
            'target_span': 0.05,  # Much smaller: ~2.9° instead of ~8.6°
            'grid_mode': 'natural'
        },
        'optimization': {'phase_iters': 500}
    }

    response = run_optimization(user_input, progress_callback=print_progress)

    if response.success:
        result = response.result.to_dict()
        print(f"\nResults: Eff={result['metrics']['total_efficiency']:.4f}, "
              f"Uniformity={result['metrics']['uniformity']:.4f}")

        # Show period info for diagnosis
        period_pixels = result.get('splitter_info', {}).get('period_pixels', 'N/A')
        print(f"Period: {period_pixels} pixels (smaller = more DOF for angle)")

        return visualize_1d_with_3x(result, "FFT 1D: Small Angle (0.05 rad)",
                                     wavelength, pixel_size), result
    else:
        print(f"ERROR: {[e.message for e in response.errors]}")
        return None, None


def test_asm_3x3():
    """Type C: ASM splitter (small target)."""
    print("\n" + "="*70)
    print("TEST: ASM 3x3 (500μm target, 5mm distance)")
    print("="*70)

    wavelength = 532e-9
    working_distance = 5e-3
    pixel_size = 1e-6
    target_span = 500e-6

    user_input = {
        'doe_type': 'splitter_2d',
        'wavelength': wavelength,
        'working_distance': working_distance,
        'device_diameter': 256e-6,
        'pixel_size': pixel_size,
        'target_spec': {
            'num_spots': [3, 3],
            'target_type': 'size',
            'target_span': [target_span, target_span],
            'grid_mode': 'natural'
        },
        'optimization': {'phase_iters': 500}
    }

    response = run_optimization(user_input, progress_callback=print_progress)

    if response.success:
        result = response.result.to_dict()
        strategy = result.get('splitter_info', {}).get('strategy', 'Unknown')
        margin = result.get('splitter_info', {}).get('target_margin_factor', 1.0)
        print(f"\nStrategy: {strategy}, Margin: {margin:.2f}")
        print(f"Results: Eff={result['metrics']['total_efficiency']:.4f}")

        return visualize_2d_standard(
            result, f"ASM 3x3: {strategy} (margin={margin:.2f})",
            wavelength, pixel_size,
            working_distance=working_distance,
            target_span=target_span,
            target_pixel_size=pixel_size,
            is_physical=True
        ), result
    else:
        print(f"ERROR: {[e.message for e in response.errors]}")
        return None, None


def test_sfr_3x3():
    """Type B: SFR splitter (large target)."""
    print("\n" + "="*70)
    print("TEST: SFR 3x3 (2mm target, 10mm distance)")
    print("="*70)

    wavelength = 532e-9
    working_distance = 10e-3
    pixel_size = 1e-6
    target_span = 2e-3

    user_input = {
        'doe_type': 'splitter_2d',
        'wavelength': wavelength,
        'working_distance': working_distance,
        'device_diameter': 256e-6,
        'pixel_size': pixel_size,
        'target_spec': {
            'num_spots': [3, 3],
            'target_type': 'size',
            'target_span': [target_span, target_span],
            'grid_mode': 'natural'
        },
        'optimization': {'phase_iters': 500}
    }

    response = run_optimization(user_input, progress_callback=print_progress)

    if response.success:
        result = response.result.to_dict()
        strategy = result.get('splitter_info', {}).get('strategy', 'Unknown')
        margin = result.get('splitter_info', {}).get('target_margin_factor', 1.0)
        print(f"\nStrategy: {strategy}, Margin: {margin:.2f}")
        print(f"Results: Eff={result['metrics']['total_efficiency']:.4f}")

        target_h = len(result['target_intensity'])
        sfr_pixel_size = (target_span * margin) / target_h

        return visualize_2d_standard(
            result, f"SFR 3x3: {strategy} (margin={margin:.2f})",
            wavelength, pixel_size,
            working_distance=working_distance,
            target_span=target_span,
            target_pixel_size=sfr_pixel_size,
            is_physical=True
        ), result
    else:
        print(f"ERROR: {[e.message for e in response.errors]}")
        return None, None


def test_strategy2_5x5():
    """Strategy 2: Periodic + Fresnel."""
    print("\n" + "="*70)
    print("TEST: Strategy 2 - Periodic+Fresnel (5x5, 10mm target)")
    print("="*70)

    wavelength = 532e-9
    working_distance = 50e-3
    pixel_size = 1e-6
    target_span = 10e-3

    user_input = {
        'doe_type': 'splitter_2d',
        'wavelength': wavelength,
        'working_distance': working_distance,
        'device_diameter': 256e-6,
        'pixel_size': pixel_size,
        'target_spec': {
            'num_spots': [5, 5],
            'target_type': 'size',
            'target_span': [target_span, target_span],
            'grid_mode': 'natural'
        },
        'optimization': {'phase_iters': 500}
    }

    response = run_optimization(user_input, progress_callback=print_progress)

    if response.success:
        result = response.result.to_dict()
        strategy = result.get('splitter_info', {}).get('strategy', 'Unknown')
        print(f"\nStrategy: {strategy}")
        print(f"Results: Eff={result['metrics']['total_efficiency']:.4f}")

        return visualize_strategy2_standard(
            result, f"Strategy 2: Periodic+Fresnel (5x5)",
            wavelength, pixel_size, working_distance, target_span
        ), result
    else:
        print(f"ERROR: {[e.message for e in response.errors]}")
        return None, None


def test_strategy2_3x3_large():
    """Strategy 2: Periodic + Fresnel with larger target."""
    print("\n" + "="*70)
    print("TEST: Strategy 2 - Periodic+Fresnel (3x3, 20mm target)")
    print("="*70)

    wavelength = 532e-9
    working_distance = 50e-3
    pixel_size = 1e-6
    target_span = 20e-3

    user_input = {
        'doe_type': 'splitter_2d',
        'wavelength': wavelength,
        'working_distance': working_distance,
        'device_diameter': 256e-6,
        'pixel_size': pixel_size,
        'target_spec': {
            'num_spots': [3, 3],
            'target_type': 'size',
            'target_span': [target_span, target_span],
            'grid_mode': 'natural'
        },
        'optimization': {'phase_iters': 500}
    }

    response = run_optimization(user_input, progress_callback=print_progress)

    if response.success:
        result = response.result.to_dict()
        strategy = result.get('splitter_info', {}).get('strategy', 'Unknown')
        print(f"\nStrategy: {strategy}")
        print(f"Results: Eff={result['metrics']['total_efficiency']:.4f}")

        return visualize_strategy2_standard(
            result, f"Strategy 2: Periodic+Fresnel (3x3, 20mm)",
            wavelength, pixel_size, working_distance, target_span
        ), result
    else:
        print(f"ERROR: {[e.message for e in response.errors]}")
        return None, None


# =============================================================================
# Main
# =============================================================================

def main():
    """Run all standardized tests."""
    print("="*70)
    print("DOE Optimizer v2.0 - Standardized Splitter Tests")
    print("="*70)
    print(f"\nDevice: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    tests = [
        # FFT 2D tests
        ("fft_2d_natural", test_fft_2d_natural),
        ("fft_2d_uniform_tol5", test_fft_2d_uniform_tol5),
        ("fft_2d_uniform_tol1", test_fft_2d_uniform_tol1),
        ("fft_2d_bigpixel_2um", test_fft_2d_bigpixel_2um),
        ("fft_2d_bigpixel_4um", test_fft_2d_bigpixel_4um),
        ("fft_2d_upsample_2x", test_fft_2d_upsample_2x),  # 2x opt + 3x analysis
        # FFT 1D tests
        ("fft_1d_natural", test_fft_1d_natural),
        ("fft_1d_upsample_2x", test_fft_1d_upsample_2x),  # 2x opt + 3x analysis
        ("fft_1d_small_angle", test_fft_1d_small_angle),  # Small angle for DOF test
        # ASM test
        ("asm_3x3", test_asm_3x3),
        # SFR test
        ("sfr_3x3", test_sfr_3x3),
        # Strategy 2 tests
        ("strategy2_5x5", test_strategy2_5x5),
        ("strategy2_3x3_large", test_strategy2_3x3_large),
    ]

    results = {}

    for name, test_fn in tests:
        fig, result = test_fn()
        if fig:
            fig.savefig(f'{output_dir}/{name}.png', dpi=150, bbox_inches='tight')
            results[name] = result
            plt.close(fig)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for name, result in results.items():
        metrics = result.get('metrics', {})
        strategy = result.get('splitter_info', {}).get('strategy', 'FFT')
        period = result.get('splitter_info', {}).get('period_pixels', 'N/A')
        print(f"\n{name}:")
        print(f"  Strategy: {strategy if strategy else 'FFT'}")
        print(f"  Period: {period}px")
        print(f"  Efficiency: {metrics.get('total_efficiency', 0):.4f}")
        print(f"  Uniformity: {metrics.get('uniformity', 0):.4f}")

    print(f"\nResults saved to '{output_dir}/'")


if __name__ == '__main__':
    main()
