/**
 * Pattern Upload Module (Auto-process Version)
 * Two-step processing (automatic):
 *   Step 1: Grayscale + user scaling (none, ratio, or width)
 *   Step 2: Crop/pad to target pixels + normalization
 */

const PatternUpload = {
    // State
    originalImage: null,
    originalData: null,
    processedPattern: null,

    /**
     * Initialize the pattern upload module
     */
    init() {
        // File input handler
        const fileInput = document.getElementById('pattern-file-input');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        }

        // Upload button
        const uploadBtn = document.getElementById('upload-pattern-btn');
        if (uploadBtn) {
            uploadBtn.addEventListener('click', () => this.open());
        }

        // Clear button
        const clearBtn = document.getElementById('clear-pattern-btn');
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clearPattern());
        }

        // Scale mode selector - auto process on change
        const scaleMode = document.getElementById('pattern-scale-mode');
        if (scaleMode) {
            scaleMode.addEventListener('change', (e) => {
                this.handleScaleModeChange(e);
                this.autoProcess();
            });
        }

        // Scale value input - auto process on change
        const scaleValue = document.getElementById('pattern-scale-value');
        if (scaleValue) {
            scaleValue.addEventListener('input', () => this.autoProcess());
        }
    },

    /**
     * Open the upload modal
     */
    open() {
        const modal = document.getElementById('pattern-upload-modal');
        if (modal) {
            modal.style.display = 'flex';
            this.reset();
            this.updateTargetPixels();
        }
    },

    /**
     * Close the upload modal
     */
    close() {
        const modal = document.getElementById('pattern-upload-modal');
        if (modal) {
            modal.style.display = 'none';
        }
    },

    /**
     * Reset state
     */
    reset() {
        this.originalImage = null;
        this.originalData = null;
        this.processedPattern = null;

        // Reset UI
        document.getElementById('pattern-filename').textContent = '';
        document.getElementById('original-size').textContent = '';
        document.getElementById('apply-pattern-btn').disabled = true;
        document.getElementById('final-size').textContent = '--';
        document.getElementById('value-range').textContent = '--';

        // Reset scale mode
        document.getElementById('pattern-scale-mode').value = 'none';
        document.getElementById('pattern-scale-value').style.display = 'none';
        document.getElementById('pattern-scale-value').value = '1.0';

        // Reset preview
        const procPreview = document.getElementById('processed-preview');
        if (procPreview) {
            procPreview.innerHTML = '<p class="placeholder-text">Select an image</p>';
        }
    },

    /**
     * Update target pixels display from DOE Settings
     */
    updateTargetPixels() {
        const simPixels = AppState.structuredParams.simulation_pixels;
        const size = Array.isArray(simPixels) ? simPixels : [simPixels, simPixels];
        document.getElementById('target-pixels').textContent = `${size[0]} x ${size[1]}`;
    },

    /**
     * Handle file selection
     */
    handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;

        document.getElementById('pattern-filename').textContent = file.name;

        // Read the image
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => {
                this.originalImage = img;
                this.loadImageData(img);
            };
            img.src = e.target.result;
        };
        reader.readAsDataURL(file);
    },

    /**
     * Load image data to canvas and extract pixel data
     */
    loadImageData(img) {
        // Create canvas to extract pixel data
        const canvas = document.createElement('canvas');
        canvas.width = img.width;
        canvas.height = img.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);

        const imageData = ctx.getImageData(0, 0, img.width, img.height);

        // Convert to grayscale 2D array
        this.originalData = this.toGrayscale(imageData);

        // Update UI
        document.getElementById('original-size').textContent = `(${img.width} x ${img.height})`;

        // Auto-process immediately
        this.autoProcess();
    },

    /**
     * Convert ImageData to grayscale 2D array
     */
    toGrayscale(imageData) {
        const width = imageData.width;
        const height = imageData.height;
        const data = imageData.data;
        const grayscale = [];

        for (let y = 0; y < height; y++) {
            const row = [];
            for (let x = 0; x < width; x++) {
                const i = (y * width + x) * 4;
                // Luminance formula
                const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
                row.push(gray / 255.0);
            }
            grayscale.push(row);
        }

        return grayscale;
    },

    /**
     * Handle scale mode change
     */
    handleScaleModeChange(event) {
        const mode = event.target.value;
        const scaleInput = document.getElementById('pattern-scale-value');

        if (mode === 'none') {
            scaleInput.style.display = 'none';
        } else {
            scaleInput.style.display = 'block';
            if (mode === 'ratio') {
                scaleInput.value = '1.0';
                scaleInput.step = '0.1';
                scaleInput.min = '0.1';
            } else if (mode === 'width') {
                scaleInput.value = this.originalData ? this.originalData[0].length : '100';
                scaleInput.step = '1';
                scaleInput.min = '1';
            }
        }
    },

    /**
     * Auto-process when data or settings change
     */
    autoProcess() {
        if (!this.originalData) return;

        const mode = document.getElementById('pattern-scale-mode').value;
        const srcH = this.originalData.length;
        const srcW = this.originalData[0].length;

        // Step 1: Scale
        let scaled;
        if (mode === 'none') {
            scaled = this.originalData.map(row => [...row]);
        } else if (mode === 'ratio') {
            const ratio = parseFloat(document.getElementById('pattern-scale-value').value) || 1.0;
            const targetH = Math.round(srcH * ratio);
            const targetW = Math.round(srcW * ratio);
            scaled = this.scaleImage(this.originalData, targetH, targetW);
        } else if (mode === 'width') {
            const targetW = parseInt(document.getElementById('pattern-scale-value').value) || srcW;
            const ratio = targetW / srcW;
            const targetH = Math.round(srcH * ratio);
            scaled = this.scaleImage(this.originalData, targetH, targetW);
        }

        // Step 2: Crop/Pad to target simulation_pixels
        const simPixels = AppState.structuredParams.simulation_pixels;
        const targetSize = Array.isArray(simPixels) ? simPixels : [simPixels, simPixels];
        const targetH = targetSize[0];
        const targetW = targetSize[1];

        let fitted = this.fitToTarget(scaled, targetH, targetW);

        // Step 2: Normalize
        const normalized = this.normalize(fitted);

        this.processedPattern = normalized;

        // Show preview
        this.showPreview();

        // Enable apply button
        document.getElementById('apply-pattern-btn').disabled = false;
    },

    /**
     * Scale image using bilinear interpolation
     */
    scaleImage(data, targetH, targetW) {
        const srcH = data.length;
        const srcW = data[0].length;

        if (srcH === targetH && srcW === targetW) {
            return data.map(row => [...row]);
        }

        const result = [];
        for (let y = 0; y < targetH; y++) {
            const row = [];
            for (let x = 0; x < targetW; x++) {
                const srcY = (y / (targetH - 1 || 1)) * (srcH - 1);
                const srcX = (x / (targetW - 1 || 1)) * (srcW - 1);

                const y0 = Math.floor(srcY);
                const y1 = Math.min(y0 + 1, srcH - 1);
                const x0 = Math.floor(srcX);
                const x1 = Math.min(x0 + 1, srcW - 1);

                const fy = srcY - y0;
                const fx = srcX - x0;

                const value = (1 - fy) * (1 - fx) * data[y0][x0] +
                              (1 - fy) * fx * data[y0][x1] +
                              fy * (1 - fx) * data[y1][x0] +
                              fy * fx * data[y1][x1];

                row.push(value);
            }
            result.push(row);
        }

        return result;
    },

    /**
     * Fit image to target size (center crop or zero-pad)
     */
    fitToTarget(data, targetH, targetW) {
        const srcH = data.length;
        const srcW = data[0].length;

        if (srcH === targetH && srcW === targetW) {
            return data.map(row => [...row]);
        }

        const result = [];
        const offsetY = Math.floor((srcH - targetH) / 2);
        const offsetX = Math.floor((srcW - targetW) / 2);

        for (let y = 0; y < targetH; y++) {
            const row = [];
            for (let x = 0; x < targetW; x++) {
                const srcY = y + offsetY;
                const srcX = x + offsetX;

                if (srcY >= 0 && srcY < srcH && srcX >= 0 && srcX < srcW) {
                    row.push(data[srcY][srcX]);
                } else {
                    row.push(0);
                }
            }
            result.push(row);
        }

        return result;
    },

    /**
     * Normalize array to [0, 1]
     */
    normalize(data) {
        let min = Infinity;
        let max = -Infinity;

        for (const row of data) {
            for (const val of row) {
                if (val < min) min = val;
                if (val > max) max = val;
            }
        }

        const range = max - min;
        if (range === 0) {
            return data.map(row => row.map(() => 1.0));
        }

        return data.map(row => row.map(val => (val - min) / range));
    },

    /**
     * Show preview plot
     */
    showPreview() {
        if (!this.processedPattern) return;

        const h = this.processedPattern.length;
        const w = this.processedPattern[0].length;

        let min = Infinity, max = -Infinity;
        for (const row of this.processedPattern) {
            for (const val of row) {
                if (val < min) min = val;
                if (val > max) max = val;
            }
        }

        Plotly.newPlot('processed-preview', [{
            z: this.processedPattern,
            type: 'heatmap',
            colorscale: 'Viridis',
            showscale: true
        }], {
            xaxis: { title: 'X', scaleanchor: 'y' },
            yaxis: { title: 'Y', autorange: 'reversed' },
            margin: { l: 40, r: 50, t: 10, b: 35 },
            width: 280,
            height: 200
        }, { responsive: true, displayModeBar: false });

        document.getElementById('final-size').textContent = `${w} x ${h}`;
        document.getElementById('value-range').textContent = `[${min.toFixed(2)}, ${max.toFixed(2)}]`;
    },

    /**
     * Apply the processed pattern to state
     */
    apply() {
        if (!this.processedPattern) return;

        AppState.targetPattern = this.processedPattern;

        const h = this.processedPattern.length;
        const w = this.processedPattern[0].length;
        AppState.structuredParams.simulation_pixels = [h, w];
        document.getElementById('param_sim_pixels').value = `[${h}, ${w}]`;

        // Update pattern status to show custom-uploaded pattern
        AppState.updatePatternStatus('custom');

        this.close();

        if (typeof Preview !== 'undefined' && Preview.update) {
            Preview.update();
        }

        console.log('Custom pattern applied:', w, 'x', h);
    },

    /**
     * Clear the custom pattern
     */
    clearPattern() {
        AppState.targetPattern = null;
        AppState.updatePatternStatus(null);

        if (typeof PreviewUI !== 'undefined' && PreviewUI.render) {
            PreviewUI.render();
        }
    }
};

window.PatternUpload = PatternUpload;
