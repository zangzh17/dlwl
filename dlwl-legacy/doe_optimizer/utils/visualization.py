"""Visualization utilities for DOE optimization results.

Provides both static (matplotlib) and interactive (plotly) visualization.

Features:
- Single-period phase visualization
- Full device phase visualization (with optional Fresnel overlay)
- Order efficiency stem plot with theoretical reference
- Scatter plot for angular/position distribution
"""

from typing import Optional, List, Tuple, Dict, Any
import numpy as np

# Import matplotlib (always available)
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import matplotlib.colors as mcolors

# Optional plotly import
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


def plot_splitter_result(
    result,  # DOEResult
    show_device: bool = True,
    show_orders: bool = True,
    figsize: Tuple[int, int] = (15, 10),
    save_path: Optional[str] = None,
    use_plotly: bool = False,
) -> Optional[Figure]:
    """Plot splitter optimization result.

    Args:
        result: DOEResult from optimize_doe()
        show_device: If True, show full device phase/height (tiled)
        show_orders: If True, show order efficiency stem plot
        figsize: Figure size for matplotlib
        save_path: Path to save figure (optional)
        use_plotly: If True, use plotly for interactive visualization

    Returns:
        matplotlib Figure if not using plotly, else None (plotly shows inline)
    """
    if use_plotly and PLOTLY_AVAILABLE:
        return _plot_splitter_plotly(result, show_device, show_orders)
    else:
        return _plot_splitter_matplotlib(result, show_device, show_orders, figsize, save_path)


def _plot_splitter_matplotlib(
    result,
    show_device: bool,
    show_orders: bool,
    figsize: Tuple[int, int],
    save_path: Optional[str],
) -> Figure:
    """Plot splitter result using matplotlib."""

    # Determine layout
    n_cols = 3 if show_orders else 2
    n_rows = 2 if show_device else 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Get splitter params for axis labels
    params = result.splitter_params
    period = params['period'] if params else None
    order_angles_deg = None
    if params and 'order_angles' in params:
        order_angles_deg = [
            (np.degrees(a[0]), np.degrees(a[1]))
            for a in params['order_angles']
        ]

    # Row 1: Period-level results
    # Phase (period)
    ax = axes[0, 0]
    im = ax.imshow(result.phase, cmap='twilight', aspect='equal')
    ax.set_title(f'Phase (Period)\nShape: {result.phase.shape}')
    plt.colorbar(im, ax=ax, label='rad')
    ax.set_xlabel('x (pixels)')
    ax.set_ylabel('y (pixels)')

    # Simulated intensity
    ax = axes[0, 1]
    im = ax.imshow(result.simulated_intensity, cmap='hot', aspect='equal')
    ax.set_title('Simulated Intensity (k-space)')
    plt.colorbar(im, ax=ax)

    # Add order index labels if available
    if params and 'order_positions' in params:
        positions = params['order_positions']
        orders = params['working_orders']
        for (py, px), (oy, ox) in zip(positions, orders):
            ax.annotate(
                f'({oy},{ox})',
                (px, py),
                color='white',
                fontsize=8,
                ha='center',
                va='center',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5)
            )

    # Order efficiency stem plot
    if show_orders and result.metrics.order_efficiencies is not None:
        ax = axes[0, 2]
        efficiencies = result.metrics.order_efficiencies
        n_orders = len(efficiencies)

        # Create labels with order indices
        # Note: evaluation may find different number of orders than config expects
        if params and 'working_orders' in params and len(params['working_orders']) == n_orders:
            labels = [f'({o[0]},{o[1]})' for o in params['working_orders']]
        else:
            labels = [str(i) for i in range(n_orders)]

        x_pos = np.arange(n_orders)
        markerline, stemlines, baseline = ax.stem(
            x_pos, efficiencies,
            linefmt='b-', markerfmt='bo', basefmt='k-'
        )

        # Add mean line
        ax.axhline(
            y=result.metrics.order_efficiency_mean,
            color='r', linestyle='--', linewidth=2,
            label=f'Mean: {result.metrics.order_efficiency_mean:.4f}'
        )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=max(6, 10 - n_orders // 10))
        ax.set_xlabel('Order Index' if len(labels) != n_orders or not params else 'Order (ny, nx)')
        ax.set_ylabel('Diffraction Efficiency')
        ax.set_title(f'Order Efficiencies ({n_orders} orders)\nUniformity: {result.metrics.order_uniformity:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Row 2: Full device representation
    if show_device and result.device_phase is not None:
        # Device phase
        ax = axes[1, 0]
        im = ax.imshow(result.device_phase, cmap='twilight', aspect='equal')
        ax.set_title(f'Device Phase (Full)\nShape: {result.device_phase.shape}')
        plt.colorbar(im, ax=ax, label='rad')

        # Add physical scale if period is known
        if period:
            ax.set_xlabel(f'x (pixels, period = {period*1e6:.1f} um)')
        else:
            ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')

        # Device height
        ax = axes[1, 1]
        im = ax.imshow(result.device_height * 1e6, cmap='viridis', aspect='equal')
        ax.set_title('Device Height (Full)')
        plt.colorbar(im, ax=ax, label='um')
        ax.set_xlabel('x (pixels)')
        ax.set_ylabel('y (pixels)')

        # Target intensity with angle labels
        if n_cols > 2:
            ax = axes[1, 2]
            im = ax.imshow(result.target_intensity, cmap='hot', aspect='equal')
            ax.set_title('Target Intensity')
            plt.colorbar(im, ax=ax)

            # Add angle labels if available
            if order_angles_deg and params and 'order_positions' in params:
                positions = params['order_positions']
                for (py, px), (ay, ax_deg) in zip(positions, order_angles_deg):
                    ax.annotate(
                        f'{ay:.1f},{ax_deg:.1f}',
                        (px, py),
                        color='white',
                        fontsize=7,
                        ha='center',
                        va='center',
                        bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.5)
                    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def _plot_splitter_plotly(
    result,
    show_device: bool,
    show_orders: bool,
) -> None:
    """Plot splitter result using plotly (interactive)."""

    if not PLOTLY_AVAILABLE:
        print("Plotly not available. Install with: pip install plotly")
        return None

    params = result.splitter_params

    # Determine number of subplots
    n_cols = 3 if show_orders else 2
    n_rows = 2 if show_device else 1

    # Create subplot titles
    titles = ['Phase (Period)', 'Simulated Intensity']
    if show_orders:
        titles.append('Order Efficiencies')
    if show_device:
        titles.extend(['Device Phase (Full)', 'Device Height (Full)'])
        if n_cols > 2:
            titles.append('Target Intensity')

    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=titles,
        specs=[[{'type': 'heatmap'}] * n_cols] * n_rows if not show_orders
              else [[{'type': 'heatmap'}, {'type': 'heatmap'}, {'type': 'xy'}]] +
                   ([[{'type': 'heatmap'}] * n_cols] if show_device else [])
    )

    # Row 1: Period-level results
    # Phase (period)
    fig.add_trace(
        go.Heatmap(
            z=result.phase,
            colorscale='twilight',
            colorbar=dict(title='rad', x=0.27),
        ),
        row=1, col=1
    )

    # Simulated intensity
    fig.add_trace(
        go.Heatmap(
            z=result.simulated_intensity,
            colorscale='hot',
            colorbar=dict(title='I', x=0.61),
        ),
        row=1, col=2
    )

    # Add order annotations
    if params and 'order_positions' in params:
        positions = params['order_positions']
        orders = params['working_orders']
        annotations = []
        for (py, px), (oy, ox) in zip(positions, orders):
            annotations.append(dict(
                x=px, y=py,
                text=f'({oy},{ox})',
                showarrow=False,
                font=dict(color='white', size=10),
                xref='x2', yref='y2'
            ))
        for ann in annotations:
            fig.add_annotation(ann)

    # Order efficiency bar chart
    if show_orders and result.metrics.order_efficiencies is not None:
        efficiencies = result.metrics.order_efficiencies

        if params and 'working_orders' in params:
            labels = [f'({o[0]},{o[1]})' for o in params['working_orders']]
        else:
            labels = [str(i) for i in range(len(efficiencies))]

        fig.add_trace(
            go.Bar(
                x=labels,
                y=efficiencies,
                marker_color='steelblue',
                name='Efficiency'
            ),
            row=1, col=3
        )

        # Add mean line
        fig.add_hline(
            y=result.metrics.order_efficiency_mean,
            line_dash='dash',
            line_color='red',
            annotation_text=f'Mean: {result.metrics.order_efficiency_mean:.4f}',
            row=1, col=3
        )

    # Row 2: Full device
    if show_device and result.device_phase is not None:
        fig.add_trace(
            go.Heatmap(
                z=result.device_phase,
                colorscale='twilight',
                colorbar=dict(title='rad', x=0.27, y=0.15, len=0.4),
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Heatmap(
                z=result.device_height * 1e6,
                colorscale='viridis',
                colorbar=dict(title='um', x=0.61, y=0.15, len=0.4),
            ),
            row=2, col=2
        )

        if n_cols > 2:
            fig.add_trace(
                go.Heatmap(
                    z=result.target_intensity,
                    colorscale='hot',
                    colorbar=dict(title='I', x=0.95, y=0.15, len=0.4),
                ),
                row=2, col=3
            )

    # Update layout
    period = params['period'] if params else None
    title = 'Splitter Optimization Result'
    if params:
        mode = params['mode'].value if hasattr(params['mode'], 'value') else str(params['mode'])
        title += f' ({mode} grid, period = {period*1e6:.1f} um)' if period else f' ({mode} grid)'

    fig.update_layout(
        title=title,
        height=400 * n_rows,
        showlegend=False,
    )

    fig.show()
    return None


def plot_order_efficiency_with_angles(
    result,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> Figure:
    """Plot order efficiencies with angle labels on x-axis.

    Args:
        result: DOEResult from optimize_doe()
        figsize: Figure size
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    params = result.splitter_params
    efficiencies = result.metrics.order_efficiencies

    if efficiencies is None:
        ax.text(0.5, 0.5, 'No order efficiency data', ha='center', va='center')
        return fig

    n_orders = len(efficiencies)

    # Get angle labels
    if params and 'order_angles' in params:
        angles = params['order_angles']
        labels = [f'{np.degrees(a[0]):.1f},{np.degrees(a[1]):.1f}' for a in angles]
        xlabel = 'Angle (theta_y, theta_x) [deg]'
    elif params and 'working_orders' in params:
        orders = params['working_orders']
        labels = [f'({o[0]},{o[1]})' for o in orders]
        xlabel = 'Order (ny, nx)'
    else:
        labels = [str(i) for i in range(n_orders)]
        xlabel = 'Order Index'

    x_pos = np.arange(n_orders)

    # Stem plot
    markerline, stemlines, baseline = ax.stem(
        x_pos, efficiencies,
        linefmt='b-', markerfmt='bo', basefmt='k-'
    )
    plt.setp(markerline, markersize=8)
    plt.setp(stemlines, linewidth=1.5)

    # Mean line
    ax.axhline(
        y=result.metrics.order_efficiency_mean,
        color='r', linestyle='--', linewidth=2,
        label=f'Mean: {result.metrics.order_efficiency_mean:.4f}'
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Diffraction Efficiency')
    ax.set_title(f'Order Efficiencies\nUniformity: {result.metrics.order_uniformity:.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_angular_distribution(
    result,
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    show_positions: bool = False,
) -> Figure:
    """Plot scatter plot of order efficiencies vs angular/position distribution.

    Args:
        result: DOEResult from optimize_doe()
        figsize: Figure size
        save_path: Path to save figure (optional)
        show_positions: If True and finite distance, show position (mm) instead of angle (deg)

    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    params = result.splitter_params
    efficiencies = result.metrics.order_efficiencies

    if efficiencies is None:
        ax.text(0.5, 0.5, 'No order efficiency data', ha='center', va='center')
        return fig

    n_orders = len(efficiencies)

    # Determine if we should show positions or angles
    z = params.get('working_distance') if params else None
    use_positions = show_positions and z is not None and params.get('target_positions')

    if use_positions:
        # Use physical positions for finite distance
        positions = params['target_positions']
        x_vals = [pos[1] * 1000 for pos in positions]  # Convert to mm
        y_vals = [pos[0] * 1000 for pos in positions]  # Convert to mm
        xlabel = 'X Position (mm)'
        ylabel = 'Y Position (mm)'
        title = f'Order Efficiencies at z = {z*1000:.1f} mm'
    elif params and 'order_angles' in params:
        # Use angles
        angles = params['order_angles']
        x_vals = [np.degrees(a[1]) for a in angles]
        y_vals = [np.degrees(a[0]) for a in angles]
        xlabel = 'X Angle (deg)'
        ylabel = 'Y Angle (deg)'
        title = 'Order Efficiencies (Angular Distribution)'
    else:
        ax.text(0.5, 0.5, 'No angle/position data', ha='center', va='center')
        return fig

    # Normalize efficiencies for color mapping
    eff_min, eff_max = min(efficiencies), max(efficiencies)
    if eff_max > eff_min:
        eff_norm = [(e - eff_min) / (eff_max - eff_min) for e in efficiencies]
    else:
        eff_norm = [1.0] * n_orders

    # Create scatter plot with efficiency as color
    scatter = ax.scatter(
        x_vals, y_vals,
        c=efficiencies,
        s=200,
        cmap='RdYlGn',
        edgecolors='black',
        linewidths=1,
        alpha=0.8
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Diffraction Efficiency')

    # Add order labels
    if params and 'working_orders' in params:
        for i, (x, y) in enumerate(zip(x_vals, y_vals)):
            order = params['working_orders'][i]
            ax.annotate(
                f'({order[0]},{order[1]})',
                (x, y),
                textcoords='offset points',
                xytext=(0, 10),
                ha='center',
                fontsize=8,
                alpha=0.7
            )

    # Add theoretical mean reference circle
    mean_eff = result.metrics.order_efficiency_mean
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f'{title}\nMean: {mean_eff:.4f}, Uniformity: {result.metrics.order_uniformity:.4f}')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_comprehensive_splitter_result(
    result,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None,
) -> Figure:
    """Plot comprehensive splitter result with all visualizations.

    Includes:
    - Single period phase
    - Full device phase (with Fresnel if applicable)
    - Order efficiency stem plot with theoretical reference
    - Angular/position scatter plot

    Args:
        result: DOEResult from optimize_doe()
        figsize: Figure size
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)

    params = result.splitter_params
    period = params['period'] if params else None
    n_spots = len(params['working_orders']) if params else 0
    z = params.get('working_distance') if params else None
    strategy = result.finite_distance_strategy

    # Title with configuration info
    mode_str = params['mode'].value if params and hasattr(params['mode'], 'value') else 'N/A'
    if z is not None:
        dist_str = f'z = {z*1000:.1f} mm'
        if strategy is not None:
            dist_str += f' (Strategy: {strategy.value})'
    else:
        dist_str = 'Infinite distance'

    fig.suptitle(
        f'Splitter Result: {mode_str.upper()} grid, {n_spots} spots\n'
        f'Period: {period*1e6:.1f} um, {dist_str}',
        fontsize=12
    )

    # Create 2x3 grid
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Row 1: Phase visualizations
    # Single period phase
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(result.phase, cmap='twilight', aspect='equal')
    ax1.set_title(f'Single Period Phase\nShape: {result.phase.shape}')
    plt.colorbar(im1, ax=ax1, label='rad')

    # Full device phase
    ax2 = fig.add_subplot(gs[0, 1])
    if result.device_phase is not None:
        im2 = ax2.imshow(result.device_phase, cmap='twilight', aspect='equal')
        ax2.set_title(f'Device Phase (Tiled)\nShape: {result.device_phase.shape}')
        plt.colorbar(im2, ax=ax2, label='rad')

    # Device phase with Fresnel (or simulated intensity if no Fresnel)
    ax3 = fig.add_subplot(gs[0, 2])
    if result.device_phase_with_fresnel is not None:
        im3 = ax3.imshow(result.device_phase_with_fresnel, cmap='twilight', aspect='equal')
        ax3.set_title('Device Phase + Fresnel')
        plt.colorbar(im3, ax=ax3, label='rad')
    else:
        im3 = ax3.imshow(result.simulated_intensity, cmap='hot', aspect='equal')
        ax3.set_title('Simulated Intensity (k-space)')
        plt.colorbar(im3, ax=ax3)

    # Row 2: Efficiency analysis
    # Stem plot with theoretical reference
    ax4 = fig.add_subplot(gs[1, 0:2])
    if result.metrics.order_efficiencies is not None:
        efficiencies = result.metrics.order_efficiencies
        n_orders = len(efficiencies)

        # Create labels
        if params and 'working_orders' in params and len(params['working_orders']) == n_orders:
            labels = [f'({o[0]},{o[1]})' for o in params['working_orders']]
        else:
            labels = [str(i) for i in range(n_orders)]

        x_pos = np.arange(n_orders)

        # Stem plot
        markerline, stemlines, baseline = ax4.stem(
            x_pos, efficiencies,
            linefmt='b-', markerfmt='bo', basefmt='k-'
        )
        plt.setp(markerline, markersize=6)

        # Theoretical reference line (1/n_spots for uniform distribution)
        theoretical_eff = 1.0 / n_spots if n_spots > 0 else 0
        ax4.axhline(
            y=theoretical_eff,
            color='green', linestyle=':', linewidth=2,
            label=f'Theoretical: {theoretical_eff:.4f}'
        )

        # Mean line
        ax4.axhline(
            y=result.metrics.order_efficiency_mean,
            color='r', linestyle='--', linewidth=2,
            label=f'Mean: {result.metrics.order_efficiency_mean:.4f}'
        )

        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(labels, rotation=45, ha='right', fontsize=max(6, 10 - n_orders // 10))
        ax4.set_xlabel('Order (ny, nx)')
        ax4.set_ylabel('Diffraction Efficiency')
        ax4.set_title(f'Order Efficiencies\nUniformity: {result.metrics.order_uniformity:.4f}, Std: {result.metrics.order_efficiency_std:.4f}')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)

    # Scatter plot
    ax5 = fig.add_subplot(gs[1, 2])
    if result.metrics.order_efficiencies is not None and params:
        efficiencies = result.metrics.order_efficiencies

        # Determine coordinates
        if z is not None and params.get('target_positions'):
            positions = params['target_positions']
            x_vals = [pos[1] * 1000 for pos in positions]
            y_vals = [pos[0] * 1000 for pos in positions]
            ax5.set_xlabel('X (mm)')
            ax5.set_ylabel('Y (mm)')
        else:
            angles = params['order_angles']
            x_vals = [np.degrees(a[1]) for a in angles]
            y_vals = [np.degrees(a[0]) for a in angles]
            ax5.set_xlabel('X Angle (deg)')
            ax5.set_ylabel('Y Angle (deg)')

        # Ensure efficiencies match positions length
        n_positions = len(x_vals)
        if len(efficiencies) != n_positions:
            # Truncate or use mean for color
            if len(efficiencies) > n_positions:
                efficiencies = efficiencies[:n_positions]
            else:
                # Use working_orders for color if lengths mismatch
                efficiencies = [result.metrics.order_efficiency_mean] * n_positions

        scatter = ax5.scatter(
            x_vals, y_vals,
            c=efficiencies,
            s=150,
            cmap='RdYlGn',
            edgecolors='black',
            linewidths=0.5,
        )
        plt.colorbar(scatter, ax=ax5, label='Efficiency')
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
        ax5.axvline(x=0, color='gray', linestyle='--', alpha=0.3)
        ax5.set_title('Spatial Distribution')
        ax5.set_aspect('equal')
        ax5.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_1d_splitter_result(
    result,
    figsize: Tuple[int, int] = (16, 10),
    save_path: Optional[str] = None,
) -> Figure:
    """Plot 1D splitter optimization result with specialized visualizations.

    For 1D splitters, uses:
    - Line plot for phase instead of 2D image
    - Bar chart with actual angle positions on x-axis
    - Proper handling of non-uniform angle spacing

    Args:
        result: DOEResult from optimize_doe()
        figsize: Figure size
        save_path: Path to save figure (optional)

    Returns:
        matplotlib Figure
    """
    fig = plt.figure(figsize=figsize)

    params = result.splitter_params
    period = params['period'] if params else None
    n_spots = len(params['working_orders']) if params else 0
    z = params.get('working_distance') if params else None
    strategy = result.finite_distance_strategy

    # Title with configuration info
    mode_str = params['mode'].value if params and hasattr(params['mode'], 'value') else 'N/A'
    if z is not None:
        dist_str = f'z = {z*1000:.1f} mm'
        if strategy is not None:
            dist_str += f' (Strategy: {strategy.value})'
    else:
        dist_str = 'Infinite distance'

    fig.suptitle(
        f'1D Splitter Result: {mode_str.upper()} grid, {n_spots} spots\n'
        f'Period: {period*1e6:.1f} um, {dist_str}',
        fontsize=12
    )

    # Create 2x2 grid
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

    # Get phase data - ensure it's 1D
    phase_data = result.phase
    if phase_data.ndim > 1:
        # Squeeze or average to get 1D
        phase_data = phase_data.squeeze()
        if phase_data.ndim > 1:
            phase_data = phase_data[:, 0]  # Take first column

    # Plot 1: Phase as line plot
    ax1 = fig.add_subplot(gs[0, 0])
    pixel_positions = np.arange(len(phase_data))
    ax1.plot(pixel_positions, phase_data, 'b-', linewidth=2)
    ax1.fill_between(pixel_positions, 0, phase_data, alpha=0.3)
    ax1.set_xlabel('Pixel Position')
    ax1.set_ylabel('Phase (rad)')
    ax1.set_title(f'Single Period Phase\n{len(phase_data)} pixels')
    ax1.set_xlim(0, len(phase_data) - 1)
    ax1.set_ylim(0, 2 * np.pi)
    ax1.axhline(y=np.pi, color='gray', linestyle='--', alpha=0.5, label='π')
    ax1.axhline(y=2*np.pi, color='gray', linestyle=':', alpha=0.5, label='2π')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Simulated intensity (k-space) as bar chart
    ax2 = fig.add_subplot(gs[0, 1])
    sim_intensity = result.simulated_intensity
    if sim_intensity.ndim > 1:
        sim_intensity = sim_intensity.squeeze()
        if sim_intensity.ndim > 1:
            sim_intensity = sim_intensity[:, 0]

    # Get order angles for x-axis
    if params and 'order_angles' in params:
        order_angles = params['order_angles']
        all_angles_deg = np.degrees([a[0] for a in order_angles])
        # Create x positions based on all pixel indices
        n_pixels = len(sim_intensity)
        center = n_pixels // 2
        all_pixel_angles = np.array([np.degrees(np.arcsin((i - center) * params['wavelength'] / period))
                                     if abs((i - center) * params['wavelength'] / period) <= 1 else np.nan
                                     for i in range(n_pixels)])
    else:
        all_pixel_angles = np.arange(len(sim_intensity)) - len(sim_intensity) // 2

    # Normalize for visualization
    sim_intensity_norm = sim_intensity / (sim_intensity.sum() + 1e-10)

    # Use uniform color (no colormap - intensity is already shown via bar height)
    bar_width = 0.8 * (all_pixel_angles[1] - all_pixel_angles[0]) if len(all_pixel_angles) > 1 else 1
    bars = ax2.bar(all_pixel_angles, sim_intensity_norm, color='steelblue', width=bar_width, edgecolor='navy', linewidth=0.5)

    ax2.set_xlabel('Angle (deg)')
    ax2.set_ylabel('Relative Intensity')
    ax2.set_title('Simulated Intensity (k-space)')
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Order efficiency bar chart with actual angles
    ax3 = fig.add_subplot(gs[1, 0])
    if result.metrics.order_efficiencies is not None:
        efficiencies = result.metrics.order_efficiencies
        n_orders = len(efficiencies)

        # Get actual angles for working orders
        if params and 'order_angles' in params:
            angles = params['order_angles']
            x_angles = [np.degrees(a[0]) for a in angles]
        else:
            x_angles = list(range(n_orders))

        # Sort by angle for proper display
        sorted_idx = np.argsort(x_angles)
        x_angles_sorted = [x_angles[i] for i in sorted_idx]
        eff_sorted = [efficiencies[i] for i in sorted_idx]

        # Bar chart with angle positions
        bar_width = (max(x_angles_sorted) - min(x_angles_sorted)) / (n_orders * 1.5) if n_orders > 1 else 1
        bars = ax3.bar(x_angles_sorted, eff_sorted, width=bar_width, color='steelblue', alpha=0.8, edgecolor='black')

        # Theoretical reference (1/n_spots)
        theoretical_eff = 1.0 / n_spots if n_spots > 0 else 0
        ax3.axhline(y=theoretical_eff, color='green', linestyle=':', linewidth=2,
                    label=f'Theoretical: {theoretical_eff:.4f}')

        # Mean line
        ax3.axhline(y=result.metrics.order_efficiency_mean, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {result.metrics.order_efficiency_mean:.4f}')

        ax3.set_xlabel('Diffraction Angle (deg)')
        ax3.set_ylabel('Diffraction Efficiency')
        ax3.set_title(f'Order Efficiencies\nUniformity: {result.metrics.order_uniformity:.4f}')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)

        # Add order labels
        if params and 'working_orders' in params:
            for i, (x_pos, eff) in enumerate(zip(x_angles_sorted, eff_sorted)):
                order = params['working_orders'][sorted_idx[i]]
                ax3.annotate(f'{order[0]}', (x_pos, eff + 0.002),
                            ha='center', va='bottom', fontsize=8)

    # Plot 4: Device phase (full) as 2D image
    ax4 = fig.add_subplot(gs[1, 1])
    if result.device_phase is not None:
        device_phase = result.device_phase

        # Ensure 2D representation for 1D splitter
        # The device_phase should be (N, M) where one dimension might be 1
        if device_phase.ndim == 1:
            # Reshape 1D to 2D for visualization (make it tall and thin)
            device_phase = device_phase.reshape(-1, 1)

        # For 1D splitter, the phase might be (N, 1) - make it visible by repeating
        if device_phase.shape[1] == 1:
            # Repeat along x to make it visible (width = height/10 or at least 50)
            repeat_factor = max(50, device_phase.shape[0] // 10)
            device_phase_2d = np.tile(device_phase, (1, repeat_factor))
        else:
            device_phase_2d = device_phase

        # Get physical dimensions
        pixel_size = params.get('pixel_size', 1e-6) if params else 1e-6
        height_um = device_phase.shape[0] * pixel_size * 1e6
        width_um = device_phase_2d.shape[1] * pixel_size * 1e6

        # Plot as 2D image with extent showing physical size
        extent = [0, width_um, height_um, 0]  # [left, right, bottom, top]
        im4 = ax4.imshow(device_phase_2d, cmap='twilight', aspect='auto', extent=extent, vmin=0, vmax=2*np.pi)
        plt.colorbar(im4, ax=ax4, label='Phase (rad)')
        ax4.set_xlabel('x (um)')
        ax4.set_ylabel('y (um)')
        ax4.set_title(f'Full Device Phase\n{device_phase.shape[0]} x {device_phase.shape[1]} pixels')

        # Add period markers
        if period:
            period_um = period * 1e6
            n_periods = int(height_um / period_um)
            for i in range(1, n_periods + 1):
                ax4.axhline(y=i * period_um, color='white', linestyle='--', alpha=0.3, linewidth=0.5)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig


def plot_finite_distance_evaluation(
    evaluation,  # FiniteDistanceEvaluation
    figsize: Tuple[int, int] = (18, 10),
    save_path: Optional[str] = None,
    title: Optional[str] = None,
) -> Figure:
    """Plot finite distance evaluation results with physical units.

    Shows:
    1. Target pattern (ideal spots)
    2. Simulated intensity at target plane with Airy circles
    3. Overlay comparison (target + simulated)
    4. Per-spot efficiency bar chart

    Args:
        evaluation: FiniteDistanceEvaluation from evaluate_finite_distance_splitter()
        figsize: Figure size
        save_path: Path to save figure (optional)
        title: Optional title override

    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Get data
    intensity = evaluation.simulated_intensity
    target = evaluation.target_intensity
    spot_positions = evaluation.spot_positions_pixels
    spot_positions_m = evaluation.spot_positions_meters
    spot_efficiencies = evaluation.spot_efficiencies
    airy_radius_px = evaluation.airy_radius_pixels
    airy_radius_m = evaluation.airy_radius_meters
    n_spots = len(spot_efficiencies)
    output_size = evaluation.output_size  # Physical size (y, x) in meters
    output_pixel_size = evaluation.output_pixel_size
    working_orders = evaluation.working_orders

    h, w = intensity.shape

    # Determine appropriate unit (mm or um)
    output_half_y = output_size[0] / 2
    output_half_x = output_size[1] / 2
    if output_half_x > 1e-3:  # > 1mm, use mm
        unit = 'mm'
        scale = 1e3
    else:  # use um
        unit = 'um'
        scale = 1e6

    # Create extent in physical units (centered)
    extent = [
        -output_half_x * scale, output_half_x * scale,
        output_half_y * scale, -output_half_y * scale
    ]

    # Airy radius in display units
    airy_radius_display = airy_radius_m * scale

    # Title
    if title is None:
        title = f'Finite Distance Evaluation\nAiry radius: {airy_radius_m*1e6:.1f} um, Output size: {output_size[0]*1e3:.1f} x {output_size[1]*1e3:.1f} mm'
    fig.suptitle(title, fontsize=12)

    # Plot 1: Target pattern
    ax1 = axes[0, 0]
    im1 = ax1.imshow(target, cmap='hot', aspect='equal', extent=extent)
    ax1.set_title('Target Pattern (Ideal Spots)')
    plt.colorbar(im1, ax=ax1, label='Intensity (norm)')
    ax1.set_xlabel(f'x ({unit})')
    ax1.set_ylabel(f'y ({unit})')
    ax1.axhline(y=0, color='white', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='white', linestyle='--', alpha=0.3)

    # Plot 2: Simulated intensity (log scale)
    ax2 = axes[0, 1]
    intensity_max = intensity.max()
    if intensity_max > 0:
        intensity_log = np.log10(intensity + intensity_max * 1e-6)
    else:
        intensity_log = np.zeros_like(intensity)
    im2 = ax2.imshow(intensity_log, cmap='hot', aspect='equal', extent=extent)
    ax2.set_title('Simulated Intensity (log scale)')
    plt.colorbar(im2, ax=ax2, label='log10(I)')
    ax2.set_xlabel(f'x ({unit})')
    ax2.set_ylabel(f'y ({unit})')

    # Draw Airy circles at target positions
    for i, (pos_y, pos_x) in enumerate(spot_positions_m):
        circle = plt.Circle(
            (pos_x * scale, pos_y * scale), airy_radius_display,
            fill=False, color='cyan', linewidth=1.0, linestyle='--'
        )
        ax2.add_patch(circle)

    # Plot 3: Overlay (target contour + simulated intensity)
    ax3 = axes[1, 0]

    # Normalize intensity for display
    if intensity_max > 0:
        intensity_norm = intensity / intensity_max
    else:
        intensity_norm = intensity

    im3 = ax3.imshow(intensity_norm, cmap='hot', aspect='equal', extent=extent)
    ax3.set_title('Simulated + Target Overlay')
    plt.colorbar(im3, ax=ax3, label='Intensity (norm)')
    ax3.set_xlabel(f'x ({unit})')
    ax3.set_ylabel(f'y ({unit})')

    # Draw Airy circles with order labels
    for i, (pos_y, pos_x) in enumerate(spot_positions_m):
        circle = plt.Circle(
            (pos_x * scale, pos_y * scale), airy_radius_display,
            fill=False, color='cyan', linewidth=1.5
        )
        ax3.add_patch(circle)

        # Label with order or efficiency
        if working_orders and i < len(working_orders):
            order = working_orders[i]
            label = f'({order[0]},{order[1]})\n{spot_efficiencies[i]:.3f}'
        else:
            label = f'{spot_efficiencies[i]:.3f}'

        ax3.annotate(
            label,
            (pos_x * scale, pos_y * scale),
            color='white', fontsize=7,
            ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.1', facecolor='black', alpha=0.6)
        )

    # Plot 4: Per-spot efficiency bar chart
    ax4 = axes[1, 1]
    x_pos = np.arange(n_spots)

    # Create labels
    if working_orders and len(working_orders) == n_spots:
        labels = [f'({o[0]},{o[1]})' for o in working_orders]
    else:
        labels = [str(i) for i in range(n_spots)]

    # Bar plot
    bars = ax4.bar(x_pos, spot_efficiencies, color='steelblue', alpha=0.8)

    # Theoretical reference (1/n)
    theoretical = 1.0 / n_spots if n_spots > 0 else 0
    ax4.axhline(y=theoretical, color='green', linestyle=':', linewidth=2,
                label=f'Theoretical: {theoretical:.4f}')

    # Mean line
    ax4.axhline(y=evaluation.mean_efficiency, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {evaluation.mean_efficiency:.4f}')

    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(labels, rotation=45, ha='right', fontsize=max(6, 10 - n_spots // 10))
    ax4.set_xlabel('Order (ny, nx)')
    ax4.set_ylabel('Efficiency (energy in Airy disk)')
    ax4.set_title(f'Spot Efficiencies\nTotal: {evaluation.total_efficiency:.4f}, Uniformity: {evaluation.uniformity:.4f}')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    return fig
