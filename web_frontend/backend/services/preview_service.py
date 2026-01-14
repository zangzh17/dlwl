"""
Preview Service for generating geometry diagrams and target pattern previews.
"""

import math
from typing import Dict, Any, Optional, Tuple, List
import numpy as np


def generate_geometry_svg(
    device_diameter: float,
    working_distance: Optional[float],
    target_span: Tuple[float, float],
    target_type: str,
    wavelength: float = 532e-9,
    doe_shape: str = "square",
    propagation_type: str = "fft"
) -> str:
    """Generate SVG geometry diagram showing side view of DOE setup.

    Args:
        device_diameter: DOE diameter in meters
        working_distance: Distance to target plane in meters (None for infinite)
        target_span: Target span (y, x) in radians or meters
        target_type: 'angle' or 'size'
        wavelength: Working wavelength in meters
        doe_shape: 'square' or 'circular'
        propagation_type: 'fft', 'asm', 'sfr', or 'periodic_fresnel'

    Returns:
        SVG string
    """
    # SVG dimensions - use viewBox for responsiveness
    # Compact layout to prevent overflow in preview column
    width, height = 350, 180
    margin = 30

    # Determine target angle
    target_angle = max(target_span) if isinstance(target_span, (list, tuple)) else target_span

    if target_type == 'angle':
        if working_distance:
            target_size = 2 * working_distance * math.tan(target_angle / 2)
        else:
            target_size = None  # Far field - no physical target size
    else:
        target_size = max(target_span) if isinstance(target_span, (list, tuple)) else target_span
        if working_distance and working_distance > 0:
            target_angle = 2 * math.atan(target_size / (2 * working_distance))

    # Center positions
    center_y = height / 2
    doe_x = margin + 20
    doe_width_px = 12

    # DOE height - fixed visual proportion (not scaled to physical size)
    doe_height = 80

    svg_parts = [
        f'<svg width="100%" height="100%" viewBox="0 0 {width} {height}" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" style="background: #fafafa;">',
        '<style>',
        '  .doe { fill: #4a90d9; stroke: #2c5aa0; stroke-width: 2; }',
        '  .beam { stroke: #ff6b6b; stroke-width: 1.5; stroke-dasharray: 4,2; opacity: 0.6; }',
        '  .beam-center { stroke: #ff6b6b; stroke-width: 2; }',
        '  .target { stroke: #2ecc71; stroke-width: 3; }',
        '  .target-fill { fill: rgba(46, 204, 113, 0.1); stroke: none; }',
        '  .annotation { font-family: Arial, sans-serif; font-size: 11px; fill: #333; }',
        '  .annotation-small { font-family: Arial, sans-serif; font-size: 9px; fill: #666; }',
        '  .dim-line { stroke: #666; stroke-width: 1; }',
        '  .dim-arrow { fill: #666; }',
        '  .axis { stroke: #ddd; stroke-width: 1; stroke-dasharray: 3,3; }',
        '</style>',
    ]

    # Optical axis
    svg_parts.append(f'<line class="axis" x1="{doe_x - 10}" y1="{center_y}" x2="{width - margin - 20}" y2="{center_y}"/>')

    # DOE element (fixed visual height)
    doe_top = center_y - doe_height / 2

    if doe_shape == 'circular':
        svg_parts.append(
            f'<ellipse class="doe" cx="{doe_x + doe_width_px/2}" cy="{center_y}" '
            f'rx="{doe_width_px/2}" ry="{doe_height/2}"/>'
        )
    else:
        svg_parts.append(
            f'<rect class="doe" x="{doe_x}" y="{doe_top}" '
            f'width="{doe_width_px}" height="{doe_height}" rx="2"/>'
        )

    # DOE label with size
    doe_size_um = device_diameter * 1e6
    if doe_size_um >= 1000:
        doe_label = f"DOE {doe_size_um/1000:.1f} mm"
    else:
        doe_label = f"DOE {doe_size_um:.0f} um"
    svg_parts.append(f'<text class="annotation" x="{doe_x + doe_width_px/2}" y="{doe_top - 8}" text-anchor="middle">{doe_label}</text>')

    if working_distance:
        # === Finite Distance Case ===
        beam_start_x = doe_x + doe_width_px
        available_x = width - margin - 80 - beam_start_x  # More room for labels
        target_x = beam_start_x + available_x * 0.75

        # Target height based on beam spread
        target_height = min(doe_height * 1.5, height - 2 * margin - 20)
        target_top = center_y - target_height / 2

        # Target plane background
        svg_parts.append(
            f'<rect class="target-fill" x="{target_x - 3}" y="{target_top}" '
            f'width="6" height="{target_height}"/>'
        )

        # Target plane line
        svg_parts.append(
            f'<line class="target" x1="{target_x}" y1="{target_top}" '
            f'x2="{target_x}" y2="{target_top + target_height}"/>'
        )

        # Target label - always show size for finite distance
        if target_size is not None:
            target_size_mm = target_size * 1e3
            if target_size_mm >= 1:
                target_label = f"Target {target_size_mm:.1f} mm"
            else:
                target_label = f"Target {target_size * 1e6:.0f} um"
        else:
            target_label = "Target"
        svg_parts.append(
            f'<text class="annotation" x="{target_x}" y="{target_top - 8}" text-anchor="middle">{target_label}</text>'
        )

        # Beam paths
        for frac in [-0.4, 0, 0.4]:
            if target_type == 'angle':
                angle = target_angle * frac
            else:
                angle = math.atan((target_size * frac) / working_distance) if working_distance > 0 and target_size else 0

            end_y = center_y + (target_x - beam_start_x) * math.tan(angle)
            beam_class = "beam-center" if frac == 0 else "beam"
            svg_parts.append(
                f'<line class="{beam_class}" x1="{beam_start_x}" y1="{center_y}" '
                f'x2="{target_x}" y2="{end_y}"/>'
            )

        # Angle annotation with arc (for finite distance too)
        if target_angle > 0:
            arc_radius = 50
            half_angle = min(target_angle / 2, 0.7)  # Limit arc for large angles
            arc_end_y = center_y - arc_radius * math.sin(half_angle)
            arc_end_x = beam_start_x + arc_radius * math.cos(half_angle)

            # Draw angle arc
            svg_parts.append(
                f'<path d="M {beam_start_x + arc_radius} {center_y} '
                f'A {arc_radius} {arc_radius} 0 0 0 {arc_end_x} {arc_end_y}" '
                f'fill="none" stroke="#666" stroke-width="1"/>'
            )

            # Angle label
            angle_deg = math.degrees(target_angle / 2)
            svg_parts.append(
                f'<text class="annotation-small" x="{beam_start_x + arc_radius + 5}" y="{center_y - 10}" '
                f'text-anchor="start">&#952; = &#177;{angle_deg:.1f}&#176;</text>'
            )

        # Working distance dimension line
        dim_y = height - 20
        wd_mm = working_distance * 1e3
        svg_parts.append(f'<line class="dim-line" x1="{beam_start_x}" y1="{dim_y}" x2="{target_x}" y2="{dim_y}"/>')
        # Arrows
        svg_parts.append(f'<polygon class="dim-arrow" points="{beam_start_x},{dim_y} {beam_start_x+6},{dim_y-3} {beam_start_x+6},{dim_y+3}"/>')
        svg_parts.append(f'<polygon class="dim-arrow" points="{target_x},{dim_y} {target_x-6},{dim_y-3} {target_x-6},{dim_y+3}"/>')
        # Dimension text
        if wd_mm >= 1:
            z_label = f"z = {wd_mm:.1f} mm"
        else:
            z_label = f"z = {working_distance * 1e6:.0f} um"
        svg_parts.append(
            f'<text class="annotation" x="{(beam_start_x + target_x)/2}" y="{dim_y - 5}" '
            f'text-anchor="middle">{z_label}</text>'
        )

    else:
        # === Infinite Distance (Far Field) Case ===
        beam_start_x = doe_x + doe_width_px
        end_x = width - margin - 30  # Leave room for labels

        # Reference plane at 100mm
        ref_distance = 0.1  # 100mm
        ref_x = beam_start_x + 120  # Fixed position for reference plane (adjusted for smaller SVG)
        ref_size = 2 * ref_distance * math.tan(target_angle / 2)  # Pattern size at 100mm
        ref_height = min(doe_height * 1.2, height - 2 * margin - 20)
        ref_top = center_y - ref_height / 2

        # Reference plane (dashed)
        svg_parts.append(
            f'<line x1="{ref_x}" y1="{ref_top}" x2="{ref_x}" y2="{ref_top + ref_height}" '
            f'stroke="#888" stroke-width="2" stroke-dasharray="8,4"/>'
        )

        # Reference plane label
        ref_size_mm = ref_size * 1e3
        svg_parts.append(
            f'<text class="annotation-small" x="{ref_x}" y="{ref_top - 5}" '
            f'text-anchor="middle">@ 100mm: {ref_size_mm:.1f}mm</text>'
        )

        # Draw diverging beam cone
        for frac in [-0.5, -0.25, 0, 0.25, 0.5]:
            angle = target_angle * frac
            travel_x = end_x - beam_start_x
            end_y = center_y + travel_x * math.tan(angle)

            # Clamp to visible area
            if end_y < margin:
                t = (margin - center_y) / (end_y - center_y) if end_y != center_y else 1
                end_x_clamped = beam_start_x + travel_x * t
                end_y = margin
            elif end_y > height - margin:
                t = (height - margin - center_y) / (end_y - center_y) if end_y != center_y else 1
                end_x_clamped = beam_start_x + travel_x * t
                end_y = height - margin
            else:
                end_x_clamped = end_x

            beam_class = "beam-center" if frac == 0 else "beam"
            svg_parts.append(
                f'<line class="{beam_class}" x1="{beam_start_x}" y1="{center_y}" '
                f'x2="{end_x_clamped}" y2="{end_y}"/>'
            )

        # Far field label (moved left to prevent overflow)
        svg_parts.append(
            f'<text class="annotation" x="{end_x - 20}" y="{margin + 15}" '
            f'text-anchor="end">Far Field (z = &#8734;)</text>'
        )

        # Angle annotation with arc (for half-angle)
        arc_radius = 60
        half_angle = target_angle / 2
        arc_end_y = center_y - arc_radius * math.sin(half_angle)
        arc_end_x = beam_start_x + arc_radius * math.cos(half_angle)

        # Draw angle arc
        svg_parts.append(
            f'<path d="M {beam_start_x + arc_radius} {center_y} '
            f'A {arc_radius} {arc_radius} 0 0 0 {arc_end_x} {arc_end_y}" '
            f'fill="none" stroke="#666" stroke-width="1"/>'
        )

        # Max angle label (half-angle with +/- notation)
        angle_deg = math.degrees(target_angle / 2)
        svg_parts.append(
            f'<text class="annotation" x="{beam_start_x + arc_radius + 15}" y="{center_y - 15}" '
            f'text-anchor="start">&#952; = &#177;{angle_deg:.2f}&#176;</text>'
        )

        # Vertical angle marker on the side
        vert_x = end_x - 30
        angle_half = target_angle / 2
        vert_travel = vert_x - beam_start_x
        vert_y_top = center_y - vert_travel * math.tan(angle_half)
        vert_y_bot = center_y + vert_travel * math.tan(angle_half)

        # Clamp to visible
        vert_y_top = max(vert_y_top, margin + 10)
        vert_y_bot = min(vert_y_bot, height - margin - 10)

        # Vertical span indicator
        svg_parts.append(
            f'<line x1="{vert_x}" y1="{vert_y_top}" x2="{vert_x}" y2="{vert_y_bot}" '
            f'stroke="#2ecc71" stroke-width="2"/>'
        )
        # Arrow heads
        svg_parts.append(
            f'<polygon fill="#2ecc71" points="{vert_x},{vert_y_top} {vert_x-4},{vert_y_top+8} {vert_x+4},{vert_y_top+8}"/>'
        )
        svg_parts.append(
            f'<polygon fill="#2ecc71" points="{vert_x},{vert_y_bot} {vert_x-4},{vert_y_bot-8} {vert_x+4},{vert_y_bot-8}"/>'
        )

    # Wavelength annotation
    wl_nm = wavelength * 1e9
    svg_parts.append(
        f'<text class="annotation-small" x="{margin}" y="{height - 8}" '
        f'text-anchor="start">lambda = {wl_nm:.0f} nm</text>'
    )

    svg_parts.append('</svg>')
    return '\n'.join(svg_parts)


def generate_target_scatter(
    working_orders: List,
    order_angles: Optional[List] = None,
    order_positions: Optional[List] = None,
    physical_positions: Optional[List] = None,
    propagation_type: str = 'fft',
    target_type: str = 'angle',
    working_distance: Optional[float] = None
) -> Optional[Dict[str, Any]]:
    """Generate scatter plot data for order positions.

    Args:
        working_orders: List of (ny, nx) order indices
        order_angles: List of (angle_y, angle_x) in radians
        order_positions: List of (py, px) pixel positions
        physical_positions: List of (y_meters, x_meters) physical positions
        propagation_type: 'fft', 'asm', 'sfr', or 'periodic_fresnel'
        target_type: 'angle' or 'size'
        working_distance: Distance to target plane in meters (for physical position calculation)

    Returns:
        Plotly-compatible scatter data, or None if no data
    """
    if not working_orders or len(working_orders) == 0:
        return None

    # Convert to lists if needed (tuples from Python)
    def to_list(item):
        if isinstance(item, (list, tuple)):
            return list(item)
        return [item, item]

    working_orders = [to_list(o) for o in working_orders]

    # Choose display based on propagation type OR availability of physical positions
    # FFT without physical_positions: angles (deg)
    # ASM/SFR/periodic_fresnel or any case with physical_positions: physical positions (mm)
    # Note: Strategy 2 (periodic_fresnel) uses FFT propagation but should show physical positions
    has_physical_positions = physical_positions and len(physical_positions) > 0
    is_physical_space = propagation_type in ('asm', 'sfr', 'periodic_fresnel') or has_physical_positions

    # Calculate physical positions from angles if not provided but working_distance is given
    if is_physical_space and not physical_positions and order_angles and working_distance:
        order_angles_list = [to_list(a) for a in order_angles]
        physical_positions = []
        for angle in order_angles_list:
            # position = working_distance * tan(angle)
            py = working_distance * math.tan(angle[0])
            px = working_distance * math.tan(angle[1])
            physical_positions.append([py, px])

    if is_physical_space and physical_positions:
        # Use physical positions for ASM/SFR
        physical_positions = [to_list(p) for p in physical_positions]
        x = [p[1] * 1e3 for p in physical_positions]  # Convert m to mm
        y = [p[0] * 1e3 for p in physical_positions]  # Convert m to mm
        x_label = 'Position X (mm)'
        y_label = 'Position Y (mm)'
    elif order_angles:
        order_angles = [to_list(a) for a in order_angles]
        # Convert to degrees for display
        x = [math.degrees(a[1]) for a in order_angles]  # angle_x
        y = [math.degrees(a[0]) for a in order_angles]  # angle_y
        x_label = 'Angle X (deg)'
        y_label = 'Angle Y (deg)'
    elif order_positions:
        order_positions = [to_list(p) for p in order_positions]
        x = [p[1] for p in order_positions]  # px
        y = [p[0] for p in order_positions]  # py
        x_label = 'Position X (pixels)'
        y_label = 'Position Y (pixels)'
    else:
        # Fall back to order indices
        x = [o[1] for o in working_orders]  # nx
        y = [o[0] for o in working_orders]  # ny
        x_label = 'Order X'
        y_label = 'Order Y'

    labels = [f"({o[0]},{o[1]})" for o in working_orders]

    return {
        'data': [{
            'type': 'scatter',
            'x': x,
            'y': y,
            'mode': 'markers',
            'text': labels,
            'hovertemplate': '%{text}<br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>',
            'marker': {
                'size': 10,
                'color': '#4a90d9',
                'line': {'width': 1, 'color': '#2c5aa0'}
            }
        }],
        'layout': {
            'title': {'text': 'Diffraction Orders', 'font': {'size': 12}},
            'xaxis': {
                'title': x_label,
                'zeroline': True,
                'zerolinecolor': '#ddd',
                'gridcolor': '#eee'
            },
            'yaxis': {
                'title': y_label,
                'zeroline': True,
                'zerolinecolor': '#ddd',
                'gridcolor': '#eee',
                'scaleanchor': 'x'
            },
            'showlegend': False,
            'margin': {'t': 40, 'r': 20, 'b': 50, 'l': 60}
        }
    }


def resample_to_uniform_angle(
    data: np.ndarray,
    wavelength: float,
    pixel_size: float,
    max_angle: Optional[float] = None
) -> Tuple[np.ndarray, float]:
    """Resample FFT output from sin(θ) space to uniform θ space.

    In FFT output, pixel index m corresponds to sin(θ) = m * λ / (N * p),
    which means θ = arcsin(m * λ / (N * p)). This is non-linear.

    This function resamples the data to a uniform angle grid.

    Args:
        data: 2D array in FFT pixel space (uniform in sin(θ))
        wavelength: Wavelength in meters
        pixel_size: DOE pixel size in meters
        max_angle: Maximum angle in radians (auto-computed if None)

    Returns:
        (resampled_data, actual_max_angle) where resampled_data is uniform in θ
    """
    from scipy import ndimage

    h, w = data.shape

    # Compute max sin(θ) that FFT can represent
    # For FFT, the range is [-N/2, N/2) * λ/(N*p) = [-λ/(2p), λ/(2p)]
    max_sin_theta = wavelength / (2 * pixel_size)

    # Clamp to valid range for arcsin
    max_sin_theta = min(max_sin_theta, 0.999)

    # Compute actual max angle
    actual_max_angle = np.arcsin(max_sin_theta)
    if max_angle is not None:
        actual_max_angle = min(actual_max_angle, max_angle)

    # Create uniform angle grid
    theta_y = np.linspace(-actual_max_angle, actual_max_angle, h)
    theta_x = np.linspace(-actual_max_angle, actual_max_angle, w)

    # For each target angle, find the corresponding pixel position in original data
    # Original: pixel m corresponds to sin(θ) = (m - N/2) * λ / (N * p)
    # So: m = sin(θ) * N * p / λ + N/2

    # Map uniform angle to pixel coordinates
    def angle_to_pixel(theta, n, wl, ps):
        sin_theta = np.sin(theta)
        # Pixel coordinate (0 to n-1, centered at n/2)
        m = sin_theta * n * ps / wl + n / 2
        return m

    # Create coordinate arrays for interpolation
    y_coords = angle_to_pixel(theta_y, h, wavelength, pixel_size)
    x_coords = angle_to_pixel(theta_x, w, wavelength, pixel_size)

    # Create 2D coordinate grid
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')

    # Resample using map_coordinates (order=1 for linear interpolation)
    resampled = ndimage.map_coordinates(data, [yy, xx], order=1, mode='constant', cval=0)

    return resampled, actual_max_angle


def generate_target_heatmap(
    target_pattern: np.ndarray,
    title: str = 'Target Pattern',
    coordinate_type: str = 'pixels',
    physical_extent: Optional[Tuple[float, float]] = None,
    angle_extent: Optional[Tuple[float, float]] = None,
    wavelength: Optional[float] = None,
    pixel_size: Optional[float] = None,
    resample_angle: bool = False
) -> Dict[str, Any]:
    """Generate heatmap data for target pattern.

    Args:
        target_pattern: 2D numpy array of target intensity
        title: Chart title
        coordinate_type: 'pixels', 'physical', or 'angle'
        physical_extent: (y_size_m, x_size_m) for physical coordinates
        angle_extent: (y_angle_rad, x_angle_rad) for angle coordinates
        wavelength: Wavelength in meters (required for angle resampling)
        pixel_size: Pixel size in meters (required for angle resampling)
        resample_angle: If True and coordinate_type='angle', resample to uniform angle space

    Returns:
        Plotly-compatible heatmap data
    """
    # Convert to intensity if needed
    if target_pattern.ndim > 2:
        target_pattern = target_pattern.squeeze()

    # Handle 1D case
    if target_pattern.ndim == 1:
        target_pattern = target_pattern.reshape(1, -1)

    # Normalize for display
    max_val = target_pattern.max()
    if max_val > 0:
        target_pattern = target_pattern / max_val

    # Resample to uniform angle space if requested (方案C)
    actual_angle_extent = angle_extent
    if resample_angle and coordinate_type == 'angle' and wavelength and pixel_size:
        max_angle = angle_extent[0] if angle_extent else None
        target_pattern, actual_max_angle = resample_to_uniform_angle(
            target_pattern, wavelength, pixel_size, max_angle
        )
        actual_angle_extent = (actual_max_angle, actual_max_angle)
        # Re-normalize after resampling
        max_val = target_pattern.max()
        if max_val > 0:
            target_pattern = target_pattern / max_val

    z = target_pattern.tolist()
    h, w = target_pattern.shape

    # Set coordinate axes based on type
    if coordinate_type == 'physical' and physical_extent:
        # Physical coordinates (mm) - linear mapping
        y_size_mm = physical_extent[0] * 1e3
        x_size_mm = physical_extent[1] * 1e3
        x_axis = {
            'title': 'X (mm)',
            'tickvals': [0, w//4, w//2, 3*w//4, w-1],
            'ticktext': [f'{-x_size_mm/2:.2f}', f'{-x_size_mm/4:.2f}', '0',
                        f'{x_size_mm/4:.2f}', f'{x_size_mm/2:.2f}']
        }
        y_axis = {
            'title': 'Y (mm)',
            'tickvals': [0, h//4, h//2, 3*h//4, h-1],
            'ticktext': [f'{-y_size_mm/2:.2f}', f'{-y_size_mm/4:.2f}', '0',
                        f'{y_size_mm/4:.2f}', f'{y_size_mm/2:.2f}'],
            'scaleanchor': 'x'
        }
    elif coordinate_type == 'angle' and actual_angle_extent:
        # Angle coordinates (degrees) - now linear after resampling
        y_angle_deg = math.degrees(actual_angle_extent[0])
        x_angle_deg = math.degrees(actual_angle_extent[1])
        x_axis = {
            'title': 'X (deg)',
            'tickvals': [0, w//4, w//2, 3*w//4, w-1],
            'ticktext': [f'{-x_angle_deg:.2f}', f'{-x_angle_deg/2:.2f}', '0',
                        f'{x_angle_deg/2:.2f}', f'{x_angle_deg:.2f}']
        }
        y_axis = {
            'title': 'Y (deg)',
            'tickvals': [0, h//4, h//2, 3*h//4, h-1],
            'ticktext': [f'{-y_angle_deg:.2f}', f'{-y_angle_deg/2:.2f}', '0',
                        f'{y_angle_deg/2:.2f}', f'{y_angle_deg:.2f}'],
            'scaleanchor': 'x'
        }
    else:
        # Default: pixel coordinates
        x_axis = {'title': 'X (pixels)'}
        y_axis = {'title': 'Y (pixels)', 'scaleanchor': 'x'}

    return {
        'data': [{
            'type': 'heatmap',
            'z': z,
            'colorscale': 'Hot',
            'showscale': True,
            'colorbar': {'title': 'Intensity', 'titleside': 'right'}
        }],
        'layout': {
            'title': {'text': title, 'font': {'size': 12}},
            'xaxis': x_axis,
            'yaxis': y_axis,
            'margin': {'t': 40, 'r': 80, 'b': 50, 'l': 60}
        }
    }
