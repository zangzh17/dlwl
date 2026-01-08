"""
Visualization module for DOE optimization results.

This module provides:
- VisualizationData: Plotly-optimized data structures
- Data export functions for frontend rendering
- create_finite_distance_visualization: For finite distance SFR evaluation
"""

from .data import (
    HeatmapData,
    BarChartData,
    ScatterData,
    LineChartData,
    VisualizationData,
    create_phase_heatmap,
    create_intensity_heatmap,
    create_efficiency_bar_chart,
    create_finite_distance_visualization,
)

__all__ = [
    'HeatmapData',
    'BarChartData',
    'ScatterData',
    'LineChartData',
    'VisualizationData',
    'create_phase_heatmap',
    'create_intensity_heatmap',
    'create_efficiency_bar_chart',
    'create_finite_distance_visualization',
]
