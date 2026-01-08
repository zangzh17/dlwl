/**
 * Preview Panel - Geometry diagram and target pattern visualization
 */

const PreviewUI = {
    /**
     * Initialize preview panel
     */
    init() {
        this.bindEvents();
    },

    /**
     * Bind event listeners
     */
    bindEvents() {
        // Target view toggle in configure panel
        document.querySelectorAll('input[name="target_view"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.renderTargetPreview(e.target.value);
            });
        });

        // Start optimization button (now in configure panel)
        const startBtn = document.getElementById('start-optimization-btn');
        if (startBtn) {
            startBtn.addEventListener('click', () => {
                OptimizerUI.startOptimization();
            });
        }
    },

    /**
     * Render the preview (called after params generation)
     */
    render() {
        this.renderGeometry();
        // Render based on currently selected view
        const selectedView = document.querySelector('input[name="target_view"]:checked');
        const viewType = selectedView ? selectedView.value : 'scatter';
        this.renderTargetPreview(viewType);
    },

    /**
     * Render geometry diagram
     */
    renderGeometry() {
        const container = document.getElementById('geometry-preview');
        if (!container) return;

        if (AppState.geometrySvg) {
            container.innerHTML = AppState.geometrySvg;
        } else {
            container.innerHTML = '<p class="hint">Generate parameters to see preview</p>';
        }
    },

    /**
     * Render target pattern preview
     * @param {string} viewType - 'scatter' or 'heatmap'
     */
    renderTargetPreview(viewType) {
        const container = document.getElementById('target-preview');
        if (!container) return;

        // Clear and prepare container
        container.innerHTML = '';

        // Get container width for explicit Plotly dimensions
        const containerWidth = container.clientWidth || 380;
        const chartHeight = 200;

        if (viewType === 'scatter' && AppState.targetScatter) {
            // Create a div for Plotly with explicit height
            const plotDiv = document.createElement('div');
            plotDiv.style.width = '100%';
            plotDiv.style.height = chartHeight + 'px';
            container.appendChild(plotDiv);

            Plotly.newPlot(
                plotDiv,
                AppState.targetScatter.data,
                {
                    ...AppState.targetScatter.layout,
                    width: containerWidth,
                    height: chartHeight,
                    autosize: false,
                    margin: { t: 30, r: 10, b: 40, l: 50 },
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent'
                },
                { responsive: true, displayModeBar: false }
            );
        } else if (viewType === 'heatmap' && AppState.targetHeatmap) {
            const plotDiv = document.createElement('div');
            plotDiv.style.width = '100%';
            plotDiv.style.height = chartHeight + 'px';
            container.appendChild(plotDiv);

            Plotly.newPlot(
                plotDiv,
                AppState.targetHeatmap.data,
                {
                    ...AppState.targetHeatmap.layout,
                    width: containerWidth,
                    height: chartHeight,
                    autosize: false,
                    margin: { t: 30, r: 10, b: 40, l: 50 },
                    paper_bgcolor: 'transparent',
                    plot_bgcolor: 'transparent'
                },
                { responsive: true, displayModeBar: false }
            );
        } else if (AppState.targetScatter) {
            // Default to scatter if available
            this.renderTargetPreview('scatter');
        } else if (AppState.targetHeatmap) {
            // Fall back to heatmap
            this.renderTargetPreview('heatmap');
        } else {
            container.innerHTML = '<p class="hint">Generate parameters to see preview</p>';
        }
    },

    /**
     * Update preview after params generation
     * Preview is driven by DOE Settings (propagation_type, target_span, target_margin)
     */
    async updatePreview() {
        try {
            // Use getPreviewRequest which includes DOE Settings for coordinate display
            const previewRequest = AppState.getPreviewRequest();
            console.log('[Preview] Request:', {
                propagation_type: previewRequest.propagation_type,
                working_distance: previewRequest.working_distance,
                target_span_m: previewRequest.target_span_m,
                target_pattern_size: previewRequest.target_pattern
                    ? `${previewRequest.target_pattern.length}x${previewRequest.target_pattern[0].length}`
                    : null
            });

            const response = await fetch('/api/preview', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(previewRequest)
            });

            if (!response.ok) {
                const errorData = await response.json();
                console.error('[Preview] HTTP Error:', response.status, errorData);
                return false;
            }

            const data = await response.json();

            if (data.success) {
                AppState.update({
                    geometrySvg: data.geometry_svg,
                    targetScatter: data.target_scatter,
                    targetHeatmap: data.target_heatmap
                });

                this.render();
                return true;
            } else {
                console.error('Preview failed:', data.error);
                return false;
            }
        } catch (err) {
            console.error('Error generating preview:', err);
            return false;
        }
    }
};

// Make available globally
window.PreviewUI = PreviewUI;
