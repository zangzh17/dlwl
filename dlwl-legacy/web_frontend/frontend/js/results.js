/**
 * Results Panel - Display optimization results and export
 */

const ResultsUI = {
    // Current view settings
    currentPhaseView: 'full',
    currentIntensityView: 'target',
    useLogScale: false,
    analysisUpsample: 2,

    /**
     * Initialize results panel
     */
    init() {
        this.bindEvents();
    },

    /**
     * Bind event listeners
     */
    bindEvents() {
        document.getElementById('new-optimization-btn').addEventListener('click', () => {
            AppState.resetOptimization();
            App.goToStep('configure');
        });

        // Chart toggles
        document.getElementById('show-efficiency-chart').addEventListener('change', (e) => {
            this.toggleChart('efficiency-chart', e.target.checked);
        });

        document.getElementById('show-order-positions').addEventListener('change', (e) => {
            this.toggleChart('order-positions-chart', e.target.checked);
        });

        document.getElementById('show-loss-history').addEventListener('change', (e) => {
            this.toggleChart('loss-history-chart', e.target.checked);
            if (e.target.checked) {
                this.renderLossHistory();
            }
        });

        // Phase view toggle
        document.querySelectorAll('input[name="phase_view"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.currentPhaseView = e.target.value;
                this.renderPhaseChart();
            });
        });

        // Intensity view toggle
        document.querySelectorAll('input[name="intensity_view"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.currentIntensityView = e.target.value;
                this.renderIntensityChart();
            });
        });

        // Log scale toggle
        document.getElementById('intensity-log-scale').addEventListener('change', (e) => {
            this.useLogScale = e.target.checked;
            this.renderIntensityChart();
        });

        // Analysis upsample - triggers re-evaluation at new resolution
        document.getElementById('analysis-upsample').addEventListener('change', async (e) => {
            this.analysisUpsample = parseInt(e.target.value);
            this.updateSimPixelsInfo();

            // Call re-evaluation API to re-propagate at new resolution
            await this.reevaluateAtResolution(this.analysisUpsample);
        });

        // Export buttons
        document.getElementById('export-phase-csv').addEventListener('click', () => {
            this.exportData('phase', 'csv');
        });

        document.getElementById('export-phase-npy').addEventListener('click', () => {
            this.exportData('phase', 'npy');
        });

        document.getElementById('export-height-csv').addEventListener('click', () => {
            this.exportData('height', 'csv');
        });
    },

    /**
     * Toggle chart visibility
     * @param {string} chartId - Chart container ID
     * @param {boolean} show - Show or hide
     */
    toggleChart(chartId, show) {
        const container = document.getElementById(chartId);
        container.style.display = show ? '' : 'none';
    },

    /**
     * Check if current propagation is periodic (FFT or periodic_fresnel)
     */
    isPeriodic() {
        const propType = AppState.structuredParams.propagation_type;
        return propType === 'fft' || propType === 'periodic_fresnel';
    },

    /**
     * Check if result is 1D
     */
    is1D() {
        const result = AppState.result;
        if (!result) return false;

        // Check simulation pixels from metadata or result
        const simPixels = result.simulation_pixels || AppState.structuredParams.simulation_pixels;
        if (Array.isArray(simPixels)) {
            return simPixels[0] === 1 || simPixels[1] === 1;
        }

        // Check if phase/simulated has 1D shape
        if (result.simulated_intensity) {
            const sim = result.simulated_intensity;
            if (Array.isArray(sim)) {
                return sim.length === 1 || (sim[0] && sim[0].length === 1);
            }
        }

        return false;
    },

    /**
     * Check if single period is available
     */
    hasSinglePeriod() {
        const result = AppState.result;
        return result && result.phase_period;
    },

    /**
     * Update simulation pixels info display
     */
    updateSimPixelsInfo() {
        const infoEl = document.getElementById('analysis-sim-pixels');
        const simPixels = AppState.structuredParams.simulation_pixels;
        if (simPixels && infoEl) {
            const upsampled = simPixels.map(p => p * this.analysisUpsample);
            infoEl.textContent = `(Sim: ${upsampled[0]} x ${upsampled[1]} pixels)`;
        }
    },

    /**
     * Check if periodic (infinite far field or strategy 2)
     */
    isPeriodic() {
        const propType = AppState.structuredParams.propagation_type;
        return propType === 'fft' || propType === 'periodic_fresnel';
    },

    /**
     * Render the results panel
     */
    render() {
        // Update single period option visibility - show for periodic cases
        const singlePeriodOption = document.getElementById('single-period-option');
        if (singlePeriodOption) {
            // Show single period option for periodic cases (FFT or periodic_fresnel)
            const showPeriod = this.isPeriodic() && this.hasSinglePeriod();
            singlePeriodOption.style.display = showPeriod ? '' : 'none';
        }

        this.updateSimPixelsInfo();
        this.renderMetrics();
        this.renderPhaseChart();
        this.renderIntensityChart();
        this.renderEfficiencyChart();
        this.renderOrderPositions();
    },

    /**
     * Render metrics summary
     */
    renderMetrics() {
        const container = document.getElementById('metrics-summary');
        const result = AppState.result;

        if (!result || !result.metrics) {
            container.innerHTML = '<p>No metrics available.</p>';
            return;
        }

        const metrics = result.metrics;

        const metricCards = [];

        if (metrics.total_efficiency !== undefined) {
            metricCards.push({
                label: 'Total Efficiency',
                value: `${(metrics.total_efficiency * 100).toFixed(2)}%`,
                warning: metrics.total_efficiency < 0.5
            });
        }

        if (metrics.uniformity !== undefined) {
            metricCards.push({
                label: 'Uniformity',
                value: `${(metrics.uniformity * 100).toFixed(2)}%`,
                warning: metrics.uniformity < 0.8
            });
        }

        if (metrics.mean_efficiency !== undefined) {
            metricCards.push({
                label: 'Mean Efficiency',
                value: `${(metrics.mean_efficiency * 100).toFixed(3)}%`
            });
        }

        if (metrics.std_efficiency !== undefined) {
            metricCards.push({
                label: 'Std Efficiency',
                value: `${(metrics.std_efficiency * 100).toFixed(3)}%`
            });
        }

        container.innerHTML = metricCards.map(m =>
            `<div class="metric-card">
                <div class="metric-label">${m.label}</div>
                <div class="metric-value ${m.warning ? 'warning' : ''}">${m.value}</div>
            </div>`
        ).join('');
    },

    /**
     * Check if phase array is effectively 1D (one dimension is 1 or very narrow)
     */
    isPhaseArray1D(phase) {
        if (!phase || !Array.isArray(phase)) return false;
        const numRows = phase.length;
        const numCols = Array.isArray(phase[0]) ? phase[0].length : 1;
        // Consider 1D if one dimension is 1 or less than 3
        return numRows <= 2 || numCols <= 2;
    },

    /**
     * Tile a 2D array to create a larger array
     * @param {Array} arr - 2D array
     * @param {number} tilesY - Number of tiles in Y direction
     * @param {number} tilesX - Number of tiles in X direction
     */
    tileArray(arr, tilesY, tilesX) {
        const resultArr = [];
        for (let ty = 0; ty < tilesY; ty++) {
            for (let row = 0; row < arr.length; row++) {
                const newRow = [];
                for (let tx = 0; tx < tilesX; tx++) {
                    newRow.push(...arr[row]);
                }
                resultArr.push(newRow);
            }
        }
        return resultArr;
    },

    /**
     * Render phase distribution chart
     */
    renderPhaseChart() {
        const container = document.getElementById('phase-chart');
        const result = AppState.result;

        if (!result || !result.phase) {
            container.innerHTML = '<p>No phase data available.</p>';
            return;
        }

        // Get physical parameters for axis labeling
        // DOE pixel is physical pixel × pixel_multiplier (for annotation display)
        // But physical device size uses ORIGINAL pixel_size (device dimensions don't change)
        const physicalPixelSize = AppState.wizardInput.pixel_size || 1e-6;  // in meters
        const physicalPixelSizeUm = physicalPixelSize * 1e6;  // Original pixel size in um
        const pixelMultiplier = AppState.optimizationSettings.pixel_multiplier || 1;
        const doePixelSizeUm = physicalPixelSizeUm * pixelMultiplier;  // Effective pixel for annotation

        // Choose data based on view
        let phase = result.phase;
        let title = 'Phase Distribution';
        let isFullDevice = true;

        // For periodic (FFT) propagation, the result.phase is one period
        // We need to tile it to show the full device
        const isPeriodic = this.isPeriodic();

        // Show/hide the single period option based on propagation type
        const singlePeriodOption = document.getElementById('single-period-option');
        if (singlePeriodOption) {
            singlePeriodOption.style.display = isPeriodic ? '' : 'none';
        }

        if (this.currentPhaseView === 'period' && isPeriodic) {
            // Single period view
            title = 'Phase Distribution (Single Period)';
            isFullDevice = false;
        } else if (isPeriodic) {
            // Full device view for periodic
            // Check if backend provided combined phase with Fresnel (Strategy 2)
            if (result.device_phase_with_fresnel) {
                phase = result.device_phase_with_fresnel;
                title = 'Phase Distribution (Full Device + Fresnel Lens)';
            } else {
                // Tile the phase manually
                const numPeriods = AppState.structuredParams.num_periods || [1, 1];
                const tilesY = Math.max(1, Math.floor(numPeriods[0]));
                const tilesX = Math.max(1, Math.floor(numPeriods[1]));

                if (tilesY > 1 || tilesX > 1) {
                    phase = this.tileArray(result.phase, tilesY, tilesX);
                    title = `Phase Distribution (Full Device: ${tilesY}×${tilesX} periods)`;
                } else {
                    title = 'Phase Distribution (Full Device = 1 Period)';
                }
            }
        } else {
            // Non-periodic (ASM/SFR) - phase is already full device
            title = 'Phase Distribution (Full Device)';
        }

        // Get array dimensions
        const numRows = phase.length;
        const numCols = Array.isArray(phase[0]) ? phase[0].length : 1;

        // Calculate physical size - device dimensions are fixed regardless of pixel_multiplier
        // Physical size = doe_pixels × ORIGINAL_pixel_size (full device)
        // or = period_pixels × ORIGINAL_pixel_size (single period)
        // Note: pixel_multiplier affects optimization resolution, not physical device size
        const periodPixels = AppState.structuredParams.period_pixels || numCols;
        const physicalSizeUm = isFullDevice
            ? (AppState.structuredParams.doe_pixels?.[0] || numCols) * physicalPixelSizeUm
            : periodPixels * physicalPixelSizeUm;

        // For axis labeling, compute effective pixel based on actual array size
        const effectivePixelSizeUm = physicalSizeUm / numCols;

        // Determine unit for display based on physical size
        let unit = 'um';
        let scale = 1;
        if (physicalSizeUm > 1000) {
            unit = 'mm';
            scale = 1e-3;
        }

        // For phase display: Full device is always 2D heatmap
        // Single period can be 1D line plot if the period is effectively 1D
        const showAs1D = !isFullDevice && this.isPhaseArray1D(phase);

        if (showAs1D) {
            // Single period with narrow dimension - show as 1D line plot
            // Extract 1D data (take the longer dimension)
            let yData;
            if (numRows >= numCols) {
                // Vertical orientation - take first column
                yData = phase.map(row => Array.isArray(row) ? row[0] : row);
            } else {
                // Horizontal orientation - take first row
                yData = Array.isArray(phase[0]) ? phase[0] : phase;
            }

            // Create x-axis in physical units
            const xData = yData.map((_, i) => i * effectivePixelSizeUm * scale);

            const trace = {
                x: xData,
                y: yData,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Phase',
                line: { color: '#4a90d9', width: 2 },
                marker: { size: 4 }
            };

            Plotly.newPlot(container, [trace], {
                title: title,
                xaxis: { title: `Position (${unit})` },
                yaxis: { title: 'Phase (rad)' },
                margin: { t: 40, r: 20, b: 50, l: 60 },
                annotations: [{
                    x: 0.02, y: 0.98, xref: 'paper', yref: 'paper',
                    text: `Pixel: ${doePixelSizeUm.toFixed(2)} um`,
                    showarrow: false, font: { size: 10, color: '#666' },
                    xanchor: 'left', yanchor: 'top'
                }]
            }, { responsive: true });
        } else {
            // 2D heatmap with physical units (always for full device)
            // Create axis arrays in physical units
            const xAxis = Array.from({ length: numCols }, (_, i) => i * effectivePixelSizeUm * scale);
            const yAxis = Array.from({ length: numRows }, (_, i) => i * effectivePixelSizeUm * scale);

            const trace = {
                z: phase,
                x: xAxis,
                y: yAxis,
                type: 'heatmap',
                colorscale: 'Viridis',
                showscale: true,
                colorbar: { title: 'Phase (rad)' }
            };

            Plotly.newPlot(container, [trace], {
                title: title,
                xaxis: { title: `X (${unit})` },
                yaxis: { title: `Y (${unit})`, scaleanchor: 'x' },
                margin: { t: 40, r: 80, b: 50, l: 60 },
                annotations: [{
                    x: 0.02, y: 0.98, xref: 'paper', yref: 'paper',
                    text: `Pixel: ${doePixelSizeUm.toFixed(2)} um`,
                    showarrow: false, font: { size: 10, color: '#666' },
                    xanchor: 'left', yanchor: 'top'
                }]
            }, { responsive: true });
        }
    },

    /**
     * Render intensity chart
     */
    renderIntensityChart() {
        const container = document.getElementById('intensity-chart');
        const result = AppState.result;
        const viewType = this.currentIntensityView;

        if (!result) {
            container.innerHTML = '<p>No intensity data available.</p>';
            return;
        }

        const is1D = this.is1D();
        const useLog = this.useLogScale;

        // Helper to apply log scale
        const applyLog = (data) => {
            if (!useLog) return data;
            if (Array.isArray(data[0])) {
                return data.map(row => row.map(v => v > 0 ? Math.log10(v + 1e-10) : -10));
            }
            return data.map(v => v > 0 ? Math.log10(v + 1e-10) : -10);
        };

        if (is1D) {
            // 1D line plot
            this.renderIntensity1D(container, viewType, useLog);
        } else if (viewType === 'comparison') {
            // Side by side comparison
            this.renderIntensityComparison(container, useLog);
        } else {
            // Single 2D view
            const data = viewType === 'target' ? result.target_intensity : result.simulated_intensity;

            if (!data) {
                container.innerHTML = '<p>No data available.</p>';
                return;
            }

            const processedData = applyLog(data);

            const trace = {
                z: processedData,
                type: 'heatmap',
                colorscale: 'Hot',
                showscale: true,
                colorbar: { title: useLog ? 'log10(Intensity)' : 'Intensity' }
            };

            Plotly.newPlot(container, [trace], {
                title: viewType === 'target' ? 'Target Intensity' : 'Simulated Intensity',
                xaxis: { title: 'X (pixels)' },
                yaxis: { title: 'Y (pixels)', scaleanchor: 'x' },
                margin: { t: 40, r: 20, b: 50, l: 60 }
            }, { responsive: true });
        }
    },

    /**
     * Render 1D intensity comparison
     */
    renderIntensity1D(container, viewType, useLog) {
        const result = AppState.result;

        // Extract 1D data
        const extract1D = (data) => {
            if (!data) return null;
            if (Array.isArray(data[0])) {
                return data.length === 1 ? data[0] : data.map(row => row[0]);
            }
            return data;
        };

        const target1D = extract1D(result.target_intensity);
        const sim1D = extract1D(result.simulated_intensity);

        const traces = [];

        // Normalize for comparison
        const normalize = (arr) => {
            const max = Math.max(...arr);
            return arr.map(v => max > 0 ? v / max : 0);
        };

        const applyLogArr = (arr) => useLog ? arr.map(v => v > 0 ? Math.log10(v + 1e-10) : -10) : arr;

        if ((viewType === 'target' || viewType === 'comparison') && target1D) {
            const yData = applyLogArr(normalize(target1D));
            traces.push({
                x: yData.map((_, i) => i),
                y: yData,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Target',
                line: { color: '#2ecc71', width: 2 },
                marker: { size: 4 }
            });
        }

        if ((viewType === 'simulated' || viewType === 'comparison') && sim1D) {
            const yData = applyLogArr(normalize(sim1D));
            traces.push({
                x: yData.map((_, i) => i),
                y: yData,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Simulated',
                line: { color: '#e74c3c', width: 2 },
                marker: { size: 4 }
            });
        }

        if (traces.length === 0) {
            container.innerHTML = '<p>No data available.</p>';
            return;
        }

        Plotly.newPlot(container, traces, {
            title: 'Intensity (1D)',
            xaxis: { title: 'Pixel' },
            yaxis: { title: useLog ? 'log10(Normalized Intensity)' : 'Normalized Intensity' },
            legend: { x: 1, xanchor: 'right', y: 1 },
            margin: { t: 40, r: 20, b: 50, l: 60 }
        }, { responsive: true });
    },

    /**
     * Render side-by-side intensity comparison
     */
    renderIntensityComparison(container, useLog) {
        const result = AppState.result;
        const traces = [];

        const applyLog = (data) => {
            if (!useLog) return data;
            return data.map(row => row.map(v => v > 0 ? Math.log10(v + 1e-10) : -10));
        };

        if (result.target_intensity) {
            traces.push({
                z: applyLog(result.target_intensity),
                type: 'heatmap',
                colorscale: 'Hot',
                showscale: false,
                xaxis: 'x',
                yaxis: 'y'
            });
        }

        if (result.simulated_intensity) {
            traces.push({
                z: applyLog(result.simulated_intensity),
                type: 'heatmap',
                colorscale: 'Hot',
                showscale: true,
                xaxis: 'x2',
                yaxis: 'y2',
                colorbar: { title: useLog ? 'log10(I)' : 'Intensity', x: 1.02 }
            });
        }

        Plotly.newPlot(container, traces, {
            title: 'Target vs Simulated',
            grid: { rows: 1, columns: 2, pattern: 'independent' },
            xaxis: { title: 'Target' },
            xaxis2: { title: 'Simulated' },
            margin: { t: 40, r: 80, b: 50, l: 60 }
        }, { responsive: true });
    },

    /**
     * Render efficiency bar chart with legends
     */
    renderEfficiencyChart() {
        const container = document.getElementById('efficiency-chart');
        const result = AppState.result;

        if (!result || !result.metrics || !result.metrics.order_efficiencies) {
            container.innerHTML = '<p>No efficiency data available.</p>';
            return;
        }

        const efficiencies = result.metrics.order_efficiencies;
        const numOrders = efficiencies.length;

        // Skip if too many orders
        if (numOrders > 200) {
            container.innerHTML = '<p>Too many orders to display (> 200).</p>';
            return;
        }

        // Calculate statistics
        const mean = efficiencies.reduce((a, b) => a + b, 0) / numOrders;
        const theoretical = 1 / numOrders;  // Ideal uniform distribution

        const trace = {
            x: efficiencies.map((_, i) => i),
            y: efficiencies.map(e => e * 100),
            type: 'bar',
            name: 'Order Efficiency',
            marker: { color: '#4a90d9' }
        };

        // Reference lines
        const shapes = [
            // Mean line
            {
                type: 'line',
                x0: -0.5,
                x1: numOrders - 0.5,
                y0: mean * 100,
                y1: mean * 100,
                line: { color: '#e74c3c', width: 2, dash: 'dash' }
            },
            // Theoretical line
            {
                type: 'line',
                x0: -0.5,
                x1: numOrders - 0.5,
                y0: theoretical * 100,
                y1: theoretical * 100,
                line: { color: '#2ecc71', width: 2, dash: 'dot' }
            }
        ];

        // Annotations for legend
        const annotations = [
            {
                x: numOrders - 1,
                y: mean * 100,
                xanchor: 'right',
                yanchor: 'bottom',
                text: `Mean: ${(mean * 100).toFixed(2)}%`,
                showarrow: false,
                font: { color: '#e74c3c', size: 10 }
            },
            {
                x: numOrders - 1,
                y: theoretical * 100,
                xanchor: 'right',
                yanchor: 'top',
                text: `Theoretical: ${(theoretical * 100).toFixed(2)}%`,
                showarrow: false,
                font: { color: '#2ecc71', size: 10 }
            }
        ];

        Plotly.newPlot(container, [trace], {
            title: 'Order Efficiencies',
            xaxis: { title: 'Order Index' },
            yaxis: { title: 'Efficiency (%)' },
            shapes: shapes,
            annotations: annotations,
            margin: { t: 40, r: 20, b: 50, l: 60 }
        }, { responsive: true });
    },

    /**
     * Render order positions chart
     */
    renderOrderPositions() {
        const container = document.getElementById('order-positions-chart');
        const result = AppState.result;

        if (!result) {
            container.innerHTML = '<p>No order position data available.</p>';
            return;
        }

        // Get order data from splitter_info or metadata
        const splitterInfo = result.splitter_info || AppState.metadata || {};
        const metrics = result.metrics || {};

        const orderPositions = splitterInfo.order_positions || [];
        const orderAngles = splitterInfo.order_angles || AppState.computedValues?.order_angles || [];
        const workingOrders = splitterInfo.working_orders || AppState.computedValues?.working_orders || [];
        const orderEfficiencies = metrics.order_efficiencies || [];

        if (workingOrders.length === 0 && orderPositions.length === 0) {
            container.innerHTML = '<p>No order position data available.</p>';
            return;
        }

        const is1D = this.is1D();

        if (is1D) {
            // 1D: show as bar chart with angle x-axis
            this.renderOrderPositions1D(container, workingOrders, orderAngles, orderEfficiencies);
        } else {
            // 2D: scatter plot
            this.renderOrderPositions2D(container, workingOrders, orderAngles, orderEfficiencies);
        }
    },

    /**
     * Render 1D order positions (bar chart)
     */
    renderOrderPositions1D(container, workingOrders, orderAngles, orderEfficiencies) {
        // Extract angles (assuming 1D is along first dimension)
        const angles = orderAngles.map(a => {
            if (Array.isArray(a)) {
                return a[0] * 180 / Math.PI;  // Convert to degrees
            }
            return a * 180 / Math.PI;
        });

        const trace = {
            x: angles,
            y: orderEfficiencies.map(e => e * 100),
            type: 'bar',
            name: 'Efficiency',
            marker: {
                color: orderEfficiencies.map(e => e),
                colorscale: 'Viridis',
                showscale: true,
                colorbar: { title: 'Efficiency', titleside: 'right' }
            },
            width: angles.length > 1 ? Math.abs(angles[1] - angles[0]) * 0.8 : 1
        };

        Plotly.newPlot(container, [trace], {
            title: 'Order Positions (1D)',
            xaxis: { title: 'Angle (deg)' },
            yaxis: { title: 'Efficiency (%)' },
            margin: { t: 40, r: 80, b: 50, l: 60 }
        }, { responsive: true });
    },

    /**
     * Render 2D order positions (scatter plot)
     */
    renderOrderPositions2D(container, workingOrders, orderAngles, orderEfficiencies) {
        // Check for physical positions (Strategy 2 or finite distance)
        const result = AppState.result;
        const splitterInfo = result.splitter_info || AppState.metadata || {};
        const physicalPositions = splitterInfo.physical_positions || [];
        const strategy = splitterInfo.strategy;

        let x, y, text, xTitle, yTitle;

        // Use physical positions for Strategy 2 (periodic_fresnel) or when available
        if (physicalPositions.length > 0 && (strategy === 'periodic_fresnel' || strategy === 'asm' || strategy === 'sfr')) {
            // Physical positions as [y, x] in meters, convert to mm
            x = physicalPositions.map(p => p[1] * 1000);  // Convert to mm
            y = physicalPositions.map(p => p[0] * 1000);  // Convert to mm
            text = workingOrders.map((o, i) =>
                `Order (${o[0]}, ${o[1]})<br>Position: (${y[i].toFixed(3)}, ${x[i].toFixed(3)}) mm<br>Eff: ${(orderEfficiencies[i] * 100 || 0).toFixed(2)}%`
            );
            xTitle = 'X Position (mm)';
            yTitle = 'Y Position (mm)';
        } else if (orderAngles.length > 0 && Array.isArray(orderAngles[0])) {
            // Angles as [theta_y, theta_x] in radians
            x = orderAngles.map(a => a[1] * 180 / Math.PI);
            y = orderAngles.map(a => a[0] * 180 / Math.PI);
            text = workingOrders.map((o, i) =>
                `Order (${o[0]}, ${o[1]})<br>Angle: (${y[i].toFixed(2)}, ${x[i].toFixed(2)}) deg<br>Eff: ${(orderEfficiencies[i] * 100 || 0).toFixed(2)}%`
            );
            xTitle = 'Angle X (deg)';
            yTitle = 'Angle Y (deg)';
        } else if (workingOrders.length > 0) {
            // Fall back to order indices
            x = workingOrders.map(o => o[1]);
            y = workingOrders.map(o => o[0]);
            text = workingOrders.map((o, i) =>
                `Order (${o[0]}, ${o[1]})<br>Eff: ${(orderEfficiencies[i] * 100 || 0).toFixed(2)}%`
            );
            xTitle = 'Order X';
            yTitle = 'Order Y';
        } else {
            container.innerHTML = '<p>No order position data available.</p>';
            return;
        }

        const trace = {
            x: x,
            y: y,
            mode: 'markers',
            type: 'scatter',
            text: text,
            hoverinfo: 'text',
            marker: {
                size: 15,
                color: orderEfficiencies.length > 0 ? orderEfficiencies : '#4a90d9',
                colorscale: 'Viridis',
                showscale: orderEfficiencies.length > 0,
                colorbar: orderEfficiencies.length > 0 ? { title: 'Efficiency', titleside: 'right' } : undefined,
                line: { width: 1, color: '#2c5aa0' }
            }
        };

        Plotly.newPlot(container, [trace], {
            title: 'Order Positions',
            xaxis: { title: xTitle, zeroline: true, zerolinecolor: '#ddd' },
            yaxis: { title: yTitle, zeroline: true, zerolinecolor: '#ddd', scaleanchor: 'x' },
            margin: { t: 40, r: 80, b: 50, l: 60 }
        }, { responsive: true });
    },

    /**
     * Render loss history chart
     */
    renderLossHistory() {
        const container = document.getElementById('loss-history-chart');

        if (AppState.lossHistory.length === 0) {
            container.innerHTML = '<p>No loss history available.</p>';
            return;
        }

        const trace = {
            x: AppState.lossHistory.map(h => h[0]),
            y: AppState.lossHistory.map(h => h[1]),
            type: 'scatter',
            mode: 'lines',
            name: 'Loss',
            line: { color: '#4a90d9', width: 2 }
        };

        Plotly.newPlot(container, [trace], {
            title: 'Loss History',
            xaxis: { title: 'Iteration' },
            yaxis: { title: 'Loss', type: 'log' },
            margin: { t: 40, r: 20, b: 50, l: 60 }
        }, { responsive: true });
    },

    /**
     * Re-evaluate optimization result at different resolution
     * @param {number} upsampleFactor - Resolution multiplier
     */
    async reevaluateAtResolution(upsampleFactor) {
        if (!AppState.taskId) {
            App.showToast('No task to re-evaluate', 'error');
            return;
        }

        // Show loading state
        const container = document.getElementById('intensity-chart');
        const originalContent = container.innerHTML;
        container.innerHTML = '<p class="loading">Re-evaluating at ' + upsampleFactor + 'x resolution...</p>';

        try {
            const response = await fetch(`/api/reevaluate/${AppState.taskId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ upsample_factor: upsampleFactor })
            });

            const data = await response.json();

            if (data.success) {
                // Update result with re-evaluated data
                if (data.simulated_intensity) {
                    AppState.result.simulated_intensity = data.simulated_intensity;
                }
                if (data.target_intensity) {
                    AppState.result.target_intensity = data.target_intensity;
                }
                if (data.phase) {
                    AppState.result.phase = data.phase;
                }
                if (data.metrics) {
                    AppState.result.metrics = data.metrics;
                }

                // Re-render all charts with new data
                this.renderMetrics();
                this.renderPhaseChart();
                this.renderIntensityChart();
                this.renderEfficiencyChart();
                this.renderOrderPositions();

                App.showToast(`Re-evaluated at ${upsampleFactor}x resolution`, 'success');
            } else {
                console.error('Re-evaluation failed:', data.error);
                App.showToast('Re-evaluation failed: ' + (data.error || 'Unknown error'), 'error');
                // Restore original content
                container.innerHTML = originalContent;
            }
        } catch (err) {
            console.error('Re-evaluation error:', err);
            App.showToast('Re-evaluation failed', 'error');
            container.innerHTML = originalContent;
        }
    },

    /**
     * Export data
     * @param {string} dataType - 'phase', 'height', or 'intensity'
     * @param {string} format - 'csv', 'npy', or 'json'
     */
    async exportData(dataType, format) {
        if (!AppState.taskId) {
            App.showToast('No result to export', 'error');
            return;
        }

        try {
            if (format === 'csv') {
                // CSV downloads directly
                const response = await fetch(`/api/export/${AppState.taskId}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ format: 'csv', data_type: dataType })
                });

                if (response.ok) {
                    const blob = await response.blob();
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `doe_${dataType}.csv`;
                    a.click();
                    URL.revokeObjectURL(url);
                    App.showToast('Export successful', 'success');
                } else {
                    App.showToast('Export failed', 'error');
                }
            } else if (format === 'npy') {
                // NPY returns base64
                const response = await fetch(`/api/export/${AppState.taskId}`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ format: 'npy', data_type: dataType })
                });

                const data = await response.json();

                if (data.success) {
                    // Decode base64 and download
                    const binary = atob(data.data);
                    const bytes = new Uint8Array(binary.length);
                    for (let i = 0; i < binary.length; i++) {
                        bytes[i] = binary.charCodeAt(i);
                    }
                    const blob = new Blob([bytes], { type: 'application/octet-stream' });
                    const url = URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = data.filename;
                    a.click();
                    URL.revokeObjectURL(url);
                    App.showToast('Export successful', 'success');
                } else {
                    App.showToast('Export failed', 'error');
                }
            }
        } catch (err) {
            console.error('Export error:', err);
            App.showToast('Export failed', 'error');
        }
    }
};

// Make available globally
window.ResultsUI = ResultsUI;
