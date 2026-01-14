/**
 * Main Application Controller
 */

const App = {
    /**
     * Initialize the application
     */
    init() {
        // Initialize all UI modules
        WizardUI.init();
        ParamsUI.init();
        PreviewUI.init();
        OptimizerUI.init();
        ResultsUI.init();
        PatternUpload.init();

        // Bind navigation
        this.bindNavigation();

        // Show initial panel
        this.goToStep('configure');

        console.log('DOE Optimizer v3.2 initialized');
    },

    /**
     * Bind navigation events
     */
    bindNavigation() {
        document.querySelectorAll('.step-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const step = e.target.dataset.step;
                this.goToStep(step);
            });
        });
    },

    /**
     * Navigate to a step
     * @param {string} step - Step name: 'configure', 'optimize', or 'results'
     */
    goToStep(step) {
        // Update state
        AppState.currentStep = step;

        // Update navigation buttons
        document.querySelectorAll('.step-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.step === step);
        });

        // Update panels
        document.querySelectorAll('.panel').forEach(panel => {
            panel.classList.remove('active');
        });

        const panel = document.getElementById(`${step}-panel`);
        if (panel) {
            panel.classList.add('active');
        }

        // Render panel-specific content
        switch (step) {
            case 'configure':
                // Already initialized, but refresh previews if we have data
                if (AppState.structuredParams) {
                    ParamsUI.render();
                    PreviewUI.render();
                }
                break;
            case 'optimize':
                OptimizerUI.render();
                break;
            case 'results':
                ResultsUI.render();
                break;
        }
    },

    /**
     * Show a toast notification
     * @param {string} message - Message to display
     * @param {string} type - 'success', 'error', 'warning', or 'info'
     */
    showToast(message, type = 'info') {
        // Remove existing toasts
        document.querySelectorAll('.toast').forEach(t => t.remove());

        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        document.body.appendChild(toast);

        // Auto-remove after 3 seconds
        setTimeout(() => {
            toast.remove();
        }, 3000);
    }
};

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    App.init();
});

// Make available globally
window.App = App;
