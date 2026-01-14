"""
Visualization data structures optimized for Plotly.js.

All data structures can be directly converted to Plotly trace format
for frontend rendering.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple, Union
import numpy as np


@dataclass
class HeatmapData:
    """Data for 2D heatmap visualization (Plotly heatmap trace).

    Attributes:
        z: 2D data array (will be converted to nested list)
        title: Plot title
        colorscale: Plotly colorscale name
        colorbar_title: Title for colorbar
        x_label: X-axis label
        y_label: Y-axis label
        x_range: Physical extent [x_min, x_max] for axis
        y_range: Physical extent [y_min, y_max] for axis
        unit: Unit string for axis labels (e.g., 'um', 'mm')
        zmin: Minimum value for color scale
        zmax: Maximum value for color scale
    """
    z: Union[np.ndarray, List[List[float]]]
    title: str = ""
    colorscale: str = "Viridis"
    colorbar_title: Optional[str] = None
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    x_range: Optional[Tuple[float, float]] = None
    y_range: Optional[Tuple[float, float]] = None
    unit: Optional[str] = None
    zmin: Optional[float] = None
    zmax: Optional[float] = None

    def to_plotly_trace(self) -> Dict[str, Any]:
        """Convert to Plotly heatmap trace format."""
        # Convert numpy array to list
        z_data = self.z.tolist() if isinstance(self.z, np.ndarray) else self.z

        trace = {
            'type': 'heatmap',
            'z': z_data,
            'colorscale': self.colorscale,
        }

        if self.colorbar_title:
            trace['colorbar'] = {'title': self.colorbar_title}

        if self.zmin is not None:
            trace['zmin'] = self.zmin
        if self.zmax is not None:
            trace['zmax'] = self.zmax

        # Add axis range if provided
        if self.x_range:
            h = len(z_data)
            trace['x0'] = self.x_range[0]
            trace['dx'] = (self.x_range[1] - self.x_range[0]) / len(z_data[0]) if z_data else 1

        if self.y_range:
            trace['y0'] = self.y_range[0]
            trace['dy'] = (self.y_range[1] - self.y_range[0]) / len(z_data) if z_data else 1

        return trace

    def to_plotly_layout(self) -> Dict[str, Any]:
        """Get Plotly layout settings for this heatmap."""
        layout = {
            'title': self.title,
        }
        if self.x_label:
            layout['xaxis'] = {'title': self.x_label}
        if self.y_label:
            layout['yaxis'] = {'title': self.y_label}
        return layout


@dataclass
class BarChartData:
    """Data for bar chart visualization (Plotly bar trace).

    Attributes:
        x: Category labels or values
        y: Bar heights
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        color: Bar color
        reference_lines: List of horizontal reference lines
            Each is {'y': value, 'label': str, 'color': str, 'dash': str}
        error_y: Error bar values (optional)
    """
    x: List[Union[str, float]]
    y: List[float]
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    color: Optional[str] = None
    reference_lines: List[Dict[str, Any]] = field(default_factory=list)
    error_y: Optional[List[float]] = None

    def to_plotly_trace(self) -> Dict[str, Any]:
        """Convert to Plotly bar trace format."""
        trace = {
            'type': 'bar',
            'x': self.x,
            'y': self.y,
        }
        if self.color:
            trace['marker'] = {'color': self.color}
        if self.error_y:
            trace['error_y'] = {'type': 'data', 'array': self.error_y}
        return trace

    def to_plotly_layout(self) -> Dict[str, Any]:
        """Get Plotly layout with reference lines."""
        layout = {
            'title': self.title,
            'xaxis': {'title': self.x_label},
            'yaxis': {'title': self.y_label},
        }
        if self.reference_lines:
            layout['shapes'] = []
            for line in self.reference_lines:
                layout['shapes'].append({
                    'type': 'line',
                    'x0': 0,
                    'x1': 1,
                    'xref': 'paper',
                    'y0': line['y'],
                    'y1': line['y'],
                    'line': {
                        'color': line.get('color', 'red'),
                        'dash': line.get('dash', 'dash'),
                    }
                })
        return layout


@dataclass
class ScatterData:
    """Data for scatter plot visualization (Plotly scatter trace).

    Attributes:
        x: X coordinates
        y: Y coordinates
        colors: Color values for each point (for colormap)
        sizes: Size values for each point
        labels: Hover text labels
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        colorbar_title: Title for colorbar
        colorscale: Plotly colorscale name
        marker_symbol: Marker symbol
    """
    x: List[float]
    y: List[float]
    colors: Optional[List[float]] = None
    sizes: Optional[List[float]] = None
    labels: Optional[List[str]] = None
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    colorbar_title: str = ""
    colorscale: str = "Viridis"
    marker_symbol: str = "circle"

    def to_plotly_trace(self) -> Dict[str, Any]:
        """Convert to Plotly scatter trace format."""
        trace = {
            'type': 'scatter',
            'mode': 'markers',
            'x': self.x,
            'y': self.y,
        }

        marker = {'symbol': self.marker_symbol}
        if self.colors:
            marker['color'] = self.colors
            marker['colorscale'] = self.colorscale
            marker['showscale'] = True
            if self.colorbar_title:
                marker['colorbar'] = {'title': self.colorbar_title}

        if self.sizes:
            marker['size'] = self.sizes

        trace['marker'] = marker

        if self.labels:
            trace['text'] = self.labels
            trace['hoverinfo'] = 'text'

        return trace

    def to_plotly_layout(self) -> Dict[str, Any]:
        """Get Plotly layout settings."""
        return {
            'title': self.title,
            'xaxis': {'title': self.x_label},
            'yaxis': {'title': self.y_label, 'scaleanchor': 'x'},
        }


@dataclass
class LineChartData:
    """Data for line chart visualization (Plotly scatter line trace).

    Attributes:
        x: X values
        y: Y values
        title: Plot title
        x_label: X-axis label
        y_label: Y-axis label
        line_color: Line color
        line_dash: Line dash style
        fill: Fill style ('tozeroy', 'tonexty', etc.)
    """
    x: List[float]
    y: List[float]
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    line_color: Optional[str] = None
    line_dash: Optional[str] = None
    fill: Optional[str] = None

    def to_plotly_trace(self) -> Dict[str, Any]:
        """Convert to Plotly scatter trace format."""
        trace = {
            'type': 'scatter',
            'mode': 'lines',
            'x': self.x,
            'y': self.y,
        }
        line = {}
        if self.line_color:
            line['color'] = self.line_color
        if self.line_dash:
            line['dash'] = self.line_dash
        if line:
            trace['line'] = line
        if self.fill:
            trace['fill'] = self.fill
        return trace


@dataclass
class VisualizationData:
    """Complete visualization data package for frontend.

    Contains all visualization data needed for a DOE optimization result,
    organized for Plotly.js rendering.

    Attributes:
        period_phase: Single period phase heatmap
        device_phase: Full device phase heatmap
        device_phase_with_fresnel: Phase with Fresnel overlay (finite distance)
        target_intensity: Target intensity heatmap
        simulated_intensity: Simulated intensity heatmap
        order_efficiency: Order efficiency bar chart (splitters)
        angular_distribution: Angular position scatter plot
        loss_history: Loss vs iteration line chart
        finite_distance_intensity: Finite distance simulation heatmap
        summary: Summary statistics dictionary
    """
    # Phase visualizations
    period_phase: Optional[HeatmapData] = None
    device_phase: Optional[HeatmapData] = None
    device_phase_with_fresnel: Optional[HeatmapData] = None

    # Intensity visualizations
    target_intensity: Optional[HeatmapData] = None
    simulated_intensity: Optional[HeatmapData] = None

    # Efficiency charts (splitters)
    order_efficiency: Optional[BarChartData] = None
    angular_distribution: Optional[ScatterData] = None

    # Training progress
    loss_history: Optional[LineChartData] = None

    # Finite distance evaluation
    finite_distance_intensity: Optional[HeatmapData] = None

    # Summary statistics
    summary: Dict[str, Any] = field(default_factory=dict)

    def to_plotly_json(self) -> Dict[str, Any]:
        """Export to complete Plotly-compatible JSON structure.

        Returns a dictionary where each key is a plot name, and the value
        contains 'data' (list of traces) and 'layout' for that plot.
        """
        result = {}

        # Helper function
        def add_plot(name: str, data_obj):
            if data_obj is not None:
                result[name] = {
                    'data': [data_obj.to_plotly_trace()],
                    'layout': data_obj.to_plotly_layout(),
                }

        # Add all plots
        add_plot('period_phase', self.period_phase)
        add_plot('device_phase', self.device_phase)
        add_plot('device_phase_with_fresnel', self.device_phase_with_fresnel)
        add_plot('target_intensity', self.target_intensity)
        add_plot('simulated_intensity', self.simulated_intensity)
        add_plot('order_efficiency', self.order_efficiency)
        add_plot('angular_distribution', self.angular_distribution)
        add_plot('loss_history', self.loss_history)
        add_plot('finite_distance_intensity', self.finite_distance_intensity)

        # Add summary
        result['summary'] = self.summary

        return result

    def get_available_plots(self) -> List[str]:
        """Get list of available plot names."""
        plots = []
        for name in ['period_phase', 'device_phase', 'device_phase_with_fresnel',
                     'target_intensity', 'simulated_intensity', 'order_efficiency',
                     'angular_distribution', 'loss_history', 'finite_distance_intensity']:
            if getattr(self, name) is not None:
                plots.append(name)
        return plots


def create_phase_heatmap(
    phase: np.ndarray,
    title: str = "Phase",
    pixel_size: float = None,
    unit: str = "um"
) -> HeatmapData:
    """Create a phase heatmap with standard settings.

    Args:
        phase: 2D phase array in radians
        title: Plot title
        pixel_size: Pixel size in meters (for axis labels)
        unit: Unit for axis labels

    Returns:
        HeatmapData configured for phase visualization
    """
    h, w = phase.shape

    # Calculate range if pixel_size provided
    x_range = None
    y_range = None
    if pixel_size:
        scale = 1e6 if unit == "um" else (1e3 if unit == "mm" else 1)
        half_w = (w * pixel_size * scale) / 2
        half_h = (h * pixel_size * scale) / 2
        x_range = (-half_w, half_w)
        y_range = (-half_h, half_h)

    return HeatmapData(
        z=phase,
        title=title,
        colorscale="Twilight",  # Cyclic colormap for phase
        colorbar_title="Phase (rad)",
        x_label=f"x ({unit})" if pixel_size else "x (pixels)",
        y_label=f"y ({unit})" if pixel_size else "y (pixels)",
        x_range=x_range,
        y_range=y_range,
        zmin=0,
        zmax=2 * np.pi,
    )


def create_intensity_heatmap(
    intensity: np.ndarray,
    title: str = "Intensity",
    log_scale: bool = False
) -> HeatmapData:
    """Create an intensity heatmap with standard settings.

    Args:
        intensity: 2D intensity array
        title: Plot title
        log_scale: Use logarithmic color scale

    Returns:
        HeatmapData configured for intensity visualization
    """
    if log_scale:
        z_data = np.log10(intensity + 1e-10)
        colorbar_title = "log10(Intensity)"
    else:
        z_data = intensity
        colorbar_title = "Intensity"

    return HeatmapData(
        z=z_data,
        title=title,
        colorscale="Hot",
        colorbar_title=colorbar_title,
    )


def create_efficiency_bar_chart(
    order_labels: List[str],
    efficiencies: List[float],
    mean_efficiency: float = None,
    title: str = "Order Efficiencies"
) -> BarChartData:
    """Create a bar chart for order efficiencies.

    Args:
        order_labels: Labels for each order (e.g., "(-1, 0)")
        efficiencies: Efficiency values for each order
        mean_efficiency: Mean efficiency for reference line
        title: Plot title

    Returns:
        BarChartData configured for efficiency visualization
    """
    ref_lines = []
    if mean_efficiency is not None:
        ref_lines.append({
            'y': mean_efficiency,
            'label': 'Mean',
            'color': 'red',
            'dash': 'dash'
        })

    return BarChartData(
        x=order_labels,
        y=efficiencies,
        title=title,
        x_label="Diffraction Order",
        y_label="Efficiency",
        reference_lines=ref_lines,
    )


def create_finite_distance_visualization(
    evaluation,  # FiniteDistanceEvaluation
    log_scale: bool = True,
) -> VisualizationData:
    """Create visualization data from FiniteDistanceEvaluation.

    Creates:
    - target_intensity: Ideal spot pattern
    - simulated_intensity: SFR propagation result
    - order_efficiency: Bar chart of spot efficiencies
    - angular_distribution: Scatter plot of spot positions

    Args:
        evaluation: FiniteDistanceEvaluation from evaluate_finite_distance_splitter
        log_scale: Use log scale for intensity

    Returns:
        VisualizationData for Plotly rendering
    """
    # Target intensity heatmap
    target_heatmap = create_intensity_heatmap(
        evaluation.target_intensity,
        title="Target Pattern (Ideal Spots)",
        log_scale=log_scale
    )

    # Simulated intensity heatmap
    simulated_heatmap = create_intensity_heatmap(
        evaluation.simulated_intensity,
        title=f"SFR Simulated Intensity",
        log_scale=log_scale
    )

    # Order efficiency bar chart
    n_spots = len(evaluation.spot_efficiencies)
    if evaluation.working_orders and len(evaluation.working_orders) == n_spots:
        labels = [f"({o[0]},{o[1]})" for o in evaluation.working_orders]
    else:
        labels = [str(i) for i in range(n_spots)]

    theoretical = 1.0 / n_spots if n_spots > 0 else 0
    ref_lines = [
        {'y': theoretical, 'label': 'Theoretical', 'color': 'green', 'dash': 'dot'},
        {'y': evaluation.mean_efficiency, 'label': 'Mean', 'color': 'red', 'dash': 'dash'},
    ]

    efficiency_chart = BarChartData(
        x=labels,
        y=evaluation.spot_efficiencies,
        title=f"Spot Efficiencies (Airy Disk Integration)\nTotal: {evaluation.total_efficiency:.4f}, Uniformity: {evaluation.uniformity:.4f}",
        x_label="Order (ny, nx)",
        y_label="Efficiency",
        reference_lines=ref_lines,
    )

    # Angular distribution scatter plot
    # Determine units (mm or um)
    output_half = max(evaluation.output_size) / 2
    if output_half > 1e-3:
        scale = 1e3
        unit = "mm"
    else:
        scale = 1e6
        unit = "um"

    x_vals = [pos[1] * scale for pos in evaluation.spot_positions_meters]
    y_vals = [pos[0] * scale for pos in evaluation.spot_positions_meters]

    scatter = ScatterData(
        x=x_vals,
        y=y_vals,
        colors=evaluation.spot_efficiencies,
        sizes=[15] * len(x_vals),
        labels=labels,
        title=f"Spot Positions at z = {evaluation.output_size[0] * 1e3:.1f} mm",
        x_label=f"X ({unit})",
        y_label=f"Y ({unit})",
        colorbar_title="Efficiency",
        colorscale="RdYlGn",
    )

    # Summary statistics
    summary = {
        'total_efficiency': evaluation.total_efficiency,
        'mean_efficiency': evaluation.mean_efficiency,
        'uniformity': evaluation.uniformity,
        'airy_radius_um': evaluation.airy_radius_meters * 1e6,
        'n_spots': n_spots,
    }

    return VisualizationData(
        target_intensity=target_heatmap,
        simulated_intensity=simulated_heatmap,
        order_efficiency=efficiency_chart,
        angular_distribution=scatter,
        finite_distance_intensity=simulated_heatmap,
        summary=summary,
    )
