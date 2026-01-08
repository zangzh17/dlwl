/**
 * Optimizer Panel - Optimization execution and progress tracking
 */

const OptimizerUI = {
    /**
     * Initialize optimizer panel
     */
    init() {
        this.bindEvents();
    },

    /**
     * Bind event listeners
     */
    bindEvents() {
        document.getElementById('cancel-optimization-btn').addEventListener('click', () => {
            this.cancelOptimization();
        });
    },

    /**
     * Start optimization
     */
    async startOptimization() {
        const btn = document.getElementById('start-optimization-btn');
        btn.disabled = true;
        btn.textContent = 'Starting...';

        try {
            AppState.resetOptimization();

            const request = AppState.getOptimizationRequest();

            const response = await fetch('/api/optimize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(request)
            });

            const data = await response.json();

            AppState.update({
                taskId: data.task_id,
                isOptimizing: true,
                lossHistory: []
            });

            // Navigate to optimize panel
            App.goToStep('optimize');

            // Connect WebSocket
            this.connectWebSocket(data.task_id);

        } catch (err) {
            console.error('Error starting optimization:', err);
            App.showToast('Failed to start optimization', 'error');
        } finally {
            btn.disabled = false;
            btn.textContent = 'Start Optimization';
        }
    },

    /**
     * Connect WebSocket for real-time progress updates
     * @param {string} taskId - Task ID
     */
    connectWebSocket(taskId) {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/api/ws/optimize/${taskId}`;

        console.log('Connecting to WebSocket:', wsUrl);

        const ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log('WebSocket connected');
        };

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            console.log('WebSocket message:', msg);

            if (msg.type === 'progress') {
                this.handleProgress(msg);
            } else if (msg.type === 'complete') {
                this.handleComplete(msg.status);
            } else if (msg.type === 'ping') {
                // Keep-alive, ignore
            }
        };

        ws.onerror = (err) => {
            console.error('WebSocket error:', err);
            // Fall back to polling
            this.startPolling(taskId);
        };

        ws.onclose = () => {
            console.log('WebSocket closed');
            // If still optimizing, fall back to polling
            if (AppState.isOptimizing) {
                this.startPolling(taskId);
            }
        };

        AppState.ws = ws;
    },

    /**
     * Start polling for progress (fallback when WebSocket fails)
     * @param {string} taskId - Task ID
     */
    startPolling(taskId) {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
        }

        this.pollInterval = setInterval(async () => {
            if (!AppState.isOptimizing) {
                clearInterval(this.pollInterval);
                return;
            }

            try {
                const response = await fetch(`/api/status/${taskId}`);
                const data = await response.json();

                if (data.progress) {
                    this.handleProgress(data.progress);
                }

                if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
                    this.handleComplete(data.status);
                    clearInterval(this.pollInterval);
                }
            } catch (err) {
                console.error('Polling error:', err);
            }
        }, 500);
    },

    /**
     * Handle progress update
     * @param {Object} progress - Progress data
     */
    handleProgress(progress) {
        AppState.progress = progress;

        // Add to loss history
        if (progress.current_iter !== undefined && progress.current_loss !== undefined) {
            AppState.lossHistory.push([progress.current_iter, progress.current_loss]);
        }

        // Update UI
        this.render();
    },

    /**
     * Handle optimization complete
     * @param {string} status - Completion status
     */
    async handleComplete(status) {
        AppState.isOptimizing = false;

        if (this.pollInterval) {
            clearInterval(this.pollInterval);
        }

        if (status === 'completed') {
            try {
                const response = await fetch(`/api/result/${AppState.taskId}`);
                const data = await response.json();

                if (data.success) {
                    AppState.update({
                        result: data.result?.result,
                        visualization: data.result?.visualization
                    });

                    App.goToStep('results');
                    App.showToast('Optimization completed!', 'success');
                } else {
                    App.showToast(data.error || 'Failed to get results', 'error');
                    App.goToStep('configure');
                }
            } catch (err) {
                console.error('Error getting results:', err);
                App.showToast('Failed to get results', 'error');
            }
        } else if (status === 'cancelled') {
            App.showToast('Optimization cancelled', 'warning');
            App.goToStep('configure');
        } else {
            App.showToast('Optimization failed', 'error');
            App.goToStep('configure');
        }
    },

    /**
     * Cancel optimization
     */
    async cancelOptimization() {
        if (!AppState.taskId) return;

        try {
            await fetch(`/api/cancel/${AppState.taskId}`, {
                method: 'POST'
            });

            App.showToast('Cancellation requested...', 'info');
        } catch (err) {
            console.error('Error cancelling:', err);
        }
    },

    /**
     * Render the optimizer panel
     */
    render() {
        const progress = AppState.progress || {};

        // Progress bar
        const progressPercent = progress.progress_percent || 0;
        document.getElementById('progress-fill').style.width = `${progressPercent}%`;

        // Progress text
        const currentIter = progress.current_iter || 0;
        const totalIters = progress.total_iters || '?';
        document.getElementById('progress-text').textContent =
            `${currentIter} / ${totalIters} iterations (${progressPercent.toFixed(1)}%)`;

        // Stats
        document.getElementById('current-loss').textContent =
            progress.current_loss !== undefined ? progress.current_loss.toExponential(4) : '--';

        document.getElementById('best-loss').textContent =
            progress.best_loss !== undefined ? progress.best_loss.toExponential(4) : '--';

        document.getElementById('eta').textContent =
            this.formatTime(progress.estimated_remaining_seconds);

        document.getElementById('speed').textContent =
            progress.iterations_per_second !== undefined ?
                `${progress.iterations_per_second.toFixed(1)} it/s` : '--';

        // Loss chart
        this.renderLossChart();
    },

    /**
     * Render loss chart
     */
    renderLossChart() {
        const container = document.getElementById('loss-chart');

        if (AppState.lossHistory.length === 0) {
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

        const layout = {
            title: '',
            xaxis: { title: 'Iteration' },
            yaxis: { title: 'Loss', type: 'log' },
            margin: { t: 20, r: 20, b: 50, l: 60 },
            autosize: true
        };

        Plotly.react(container, [trace], layout, { responsive: true });
    },

    /**
     * Format time in seconds to human readable string
     * @param {number} seconds - Time in seconds
     * @returns {string} Formatted time
     */
    formatTime(seconds) {
        if (seconds === undefined || seconds === null) return '--';
        if (seconds < 60) return `${Math.round(seconds)}s`;
        if (seconds < 3600) {
            const mins = Math.floor(seconds / 60);
            const secs = Math.round(seconds % 60);
            return `${mins}m ${secs}s`;
        }
        const hours = Math.floor(seconds / 3600);
        const mins = Math.round((seconds % 3600) / 60);
        return `${hours}h ${mins}m`;
    }
};

// Make available globally
window.OptimizerUI = OptimizerUI;
