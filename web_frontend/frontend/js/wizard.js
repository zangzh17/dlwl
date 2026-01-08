/**
 * Wizard Panel - DOE configuration form handling
 */

const WizardUI = {
    /**
     * Initialize wizard panel
     */
    init() {
        this.bindEvents();
        this.renderTargetSpec();
    },

    /**
     * Bind event listeners
     */
    bindEvents() {
        // DOE type change
        document.querySelectorAll('input[name="doe_type"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                AppState.wizardInput.doe_type = e.target.value;
                this.renderTargetSpec();
            });
        });

        // Global parameters
        document.getElementById('wavelength').addEventListener('change', (e) => {
            AppState.wizardInput.wavelength = parseFloat(e.target.value) * 1e-9;
        });

        document.getElementById('device_diameter').addEventListener('change', (e) => {
            AppState.wizardInput.device_diameter = parseFloat(e.target.value) * 1e-6;
        });

        document.getElementById('pixel_size').addEventListener('change', (e) => {
            AppState.wizardInput.pixel_size = parseFloat(e.target.value) * 1e-6;
        });

        document.getElementById('device_shape').addEventListener('change', (e) => {
            AppState.wizardInput.device_shape = e.target.value;
        });

        // Optimization settings
        const optMethod = document.getElementById('opt_method');
        if (optMethod) {
            optMethod.addEventListener('change', (e) => {
                AppState.wizardInput.optimization.phase_method = e.target.value;
            });
        }

        const phaseIters = document.getElementById('phase_iters');
        if (phaseIters) {
            phaseIters.addEventListener('change', (e) => {
                AppState.wizardInput.optimization.phase_iters = parseInt(e.target.value);
            });
        }

        const lossType = document.getElementById('loss_type');
        if (lossType) {
            lossType.addEventListener('change', (e) => {
                AppState.wizardInput.optimization.loss_type = e.target.value;
            });
        }

        // Collapsible sections
        document.querySelectorAll('.collapsible-header').forEach(header => {
            header.addEventListener('click', () => {
                header.parentElement.classList.toggle('open');
            });
        });

        // Generate params button
        document.getElementById('generate-params-btn').addEventListener('click', () => {
            this.generateParams();
        });
    },

    /**
     * Render target specification form based on DOE type
     */
    renderTargetSpec() {
        const container = document.getElementById('target-spec-form');
        const doeType = AppState.wizardInput.doe_type;

        let html = '';

        if (doeType.includes('splitter')) {
            html = this.renderSplitterSpec();
        } else if (doeType === 'diffuser') {
            html = this.renderDiffuserSpec();
        } else if (doeType === 'lens') {
            html = this.renderLensSpec();
        } else {
            html = this.renderCustomSpec();
        }

        container.innerHTML = html;
        this.bindTargetSpecEvents();
    },

    /**
     * Render splitter target specification
     */
    renderSplitterSpec() {
        const spec = AppState.wizardInput.target_spec;
        const is2D = AppState.wizardInput.doe_type === 'splitter_2d';

        // Convert internal rad to display deg, or m to mm
        const displaySpan = this.formatSpanForDisplay(spec.target_span, spec.target_type, is2D);

        // Grid Mode is only applicable for:
        // 1. Infinite distance (target_type = 'angle')
        // 2. Finite distance with Strategy 2 (use_strategy2 = true)
        // For finite distance without Strategy 2 (ASM/SFR), hide Grid Mode
        const isFiniteWithoutStrategy2 = spec.target_type === 'size' && !spec.use_strategy2;
        const showGridMode = !isFiniteWithoutStrategy2;

        return `
            <div class="form-grid">
                <label class="form-label">
                    ${is2D ? 'Number of Spots [Y, X]' : 'Number of Spots'}
                    <input type="text" id="num_spots"
                           value="${is2D ? JSON.stringify(spec.num_spots) : spec.num_spots}"
                           placeholder="${is2D ? '[5, 5]' : '5'}">
                </label>
                <label class="form-label">
                    Target Type
                    <select id="target_type">
                        <option value="angle" ${spec.target_type === 'angle' ? 'selected' : ''}>Angle (deg)</option>
                        <option value="size" ${spec.target_type === 'size' ? 'selected' : ''}>Physical Size (mm)</option>
                    </select>
                </label>
                <label class="form-label" id="target_span_label">
                    Target Span <span id="target_span_unit">${spec.target_type === 'angle' ? '(deg)' : '(mm)'}</span>
                    <input type="text" id="target_span"
                           value="${displaySpan}"
                           placeholder="${is2D ? '[5.7, 5.7]' : '5.7'}">
                </label>
                <label class="form-label" id="grid_mode_row" style="${showGridMode ? '' : 'display:none'}">
                    Grid Mode
                    <select id="grid_mode">
                        <option value="natural" ${spec.grid_mode === 'natural' ? 'selected' : ''}>Natural (k-space uniform)</option>
                        <option value="uniform" ${spec.grid_mode === 'uniform' ? 'selected' : ''}>Uniform (angle-space uniform)</option>
                    </select>
                </label>
            </div>
            <div class="form-grid" id="tolerance-row" style="${showGridMode && spec.grid_mode === 'uniform' ? '' : 'display:none'}">
                <label class="form-label">
                    Tolerance (%)
                    <input type="number" id="tolerance" value="${(spec.tolerance || 0.05) * 100}" step="0.1" min="0.1">
                </label>
            </div>
            <div class="form-grid" id="working-distance-row" style="${spec.target_type === 'size' ? '' : 'display:none'}">
                <label class="form-label">
                    Working Distance (mm)
                    <input type="number" id="working_distance"
                           value="${AppState.wizardInput.working_distance ? AppState.wizardInput.working_distance * 1e3 : ''}"
                           placeholder="e.g., 100">
                </label>
                <label class="form-label checkbox-inline">
                    <input type="checkbox" id="use_strategy2" ${spec.use_strategy2 ? 'checked' : ''}>
                    Use Strategy 2 (Periodic + Fresnel lens)
                </label>
            </div>
        `;
    },

    /**
     * Format span value for display (rad->deg or m->mm)
     */
    formatSpanForDisplay(span, targetType, is2D) {
        // Handle undefined/null/NaN span
        if (span == null || (typeof span === 'number' && isNaN(span))) {
            return is2D ? '[5.73, 5.73]' : '5.73';  // Default value
        }

        if (targetType === 'angle') {
            // rad to deg
            if (is2D && Array.isArray(span)) {
                const converted = span.map(v => {
                    const deg = v * 180 / Math.PI;
                    return isNaN(deg) ? 5.73 : +deg.toFixed(2);
                });
                return JSON.stringify(converted);
            }
            const deg = span * 180 / Math.PI;
            return isNaN(deg) ? '5.73' : deg.toFixed(2);
        } else {
            // m to mm
            if (is2D && Array.isArray(span)) {
                const converted = span.map(v => {
                    const mm = v * 1e3;
                    return isNaN(mm) ? 10 : +mm.toFixed(2);
                });
                return JSON.stringify(converted);
            }
            const mm = span * 1e3;
            return isNaN(mm) ? '10' : mm.toFixed(2);
        }
    },

    /**
     * Parse span value from display (deg->rad or mm->m)
     */
    parseSpanFromDisplay(val, targetType) {
        // Try to parse as JSON array first
        try {
            const parsed = JSON.parse(val);
            if (Array.isArray(parsed)) {
                const converted = parsed.map(v => {
                    const num = parseFloat(v);
                    if (isNaN(num)) return null;
                    if (targetType === 'angle') {
                        return num * Math.PI / 180;
                    } else {
                        return num * 1e-3;
                    }
                });
                // Check if any value is invalid
                if (converted.some(v => v === null)) {
                    return null;
                }
                return converted;
            }
            // Parsed but not array - treat as single number
            const num = parseFloat(parsed);
            if (isNaN(num)) return null;
            if (targetType === 'angle') {
                return num * Math.PI / 180;
            } else {
                return num * 1e-3;
            }
        } catch {
            // Try to parse as single number
            const num = parseFloat(val);
            if (isNaN(num)) return null;
            if (targetType === 'angle') {
                return num * Math.PI / 180;
            } else {
                return num * 1e-3;
            }
        }
    },

    /**
     * Render diffuser target specification
     */
    renderDiffuserSpec() {
        const spec = AppState.wizardInput.target_spec;

        // Convert internal units to display units
        const displaySpan = this.formatSpanForDisplay(spec.target_span, spec.target_type, false);

        return `
            <div class="form-grid">
                <label class="form-label">
                    Shape
                    <select id="diffuser_shape">
                        <option value="square" ${spec.shape === 'square' ? 'selected' : ''}>Square</option>
                        <option value="circular" ${spec.shape === 'circular' ? 'selected' : ''}>Circular</option>
                    </select>
                </label>
                <label class="form-label">
                    Target Type
                    <select id="target_type">
                        <option value="angle" ${spec.target_type === 'angle' ? 'selected' : ''}>Angle (deg)</option>
                        <option value="size" ${spec.target_type === 'size' ? 'selected' : ''}>Physical Size (mm)</option>
                    </select>
                </label>
                <label class="form-label" id="diffuser_span_label">
                    ${spec.target_type === 'angle' ? 'Diffusion Angle (deg)' : 'Target Size (mm)'}
                    <input type="number" id="target_span" value="${displaySpan}" step="0.1">
                </label>
            </div>
            <div class="form-grid" id="working-distance-row" style="${spec.target_type === 'size' ? '' : 'display:none'}">
                <label class="form-label">
                    Working Distance (mm)
                    <input type="number" id="working_distance"
                           value="${AppState.wizardInput.working_distance ? AppState.wizardInput.working_distance * 1e3 : ''}">
                </label>
            </div>
        `;
    },

    /**
     * Render lens target specification
     */
    renderLensSpec() {
        const spec = AppState.wizardInput.target_spec;

        // Convert from internal meters to display mm
        const focalLengthMm = spec.focal_length ? spec.focal_length * 1e3 : 10;

        return `
            <div class="form-grid">
                <label class="form-label">
                    Focal Length (mm)
                    <input type="number" id="focal_length" value="${focalLengthMm}" step="0.1" min="0.1">
                </label>
                <label class="form-label">
                    Lens Type
                    <select id="lens_type">
                        <option value="normal" ${spec.lens_type === 'normal' ? 'selected' : ''}>Normal</option>
                        <option value="cylindrical_x" ${spec.lens_type === 'cylindrical_x' ? 'selected' : ''}>Cylindrical (X)</option>
                        <option value="cylindrical_y" ${spec.lens_type === 'cylindrical_y' ? 'selected' : ''}>Cylindrical (Y)</option>
                    </select>
                </label>
                <label class="form-label">
                    Array Size [Y, X] (optional)
                    <input type="text" id="array_size" value="${spec.array_size ? JSON.stringify(spec.array_size) : ''}"
                           placeholder="[1, 1] for single lens">
                </label>
            </div>
        `;
    },

    /**
     * Render custom pattern specification
     */
    renderCustomSpec() {
        return `
            <div class="form-grid">
                <label class="form-label">
                    Target Image
                    <input type="file" id="target_image" accept="image/*">
                </label>
                <label class="form-label">
                    Target Resolution [H, W]
                    <input type="text" id="target_resolution" value="[256, 256]">
                </label>
            </div>
            <p class="note">Upload a grayscale image representing the target intensity pattern.</p>
        `;
    },

    /**
     * Update Grid Mode visibility based on target_type and use_strategy2
     */
    updateGridModeVisibility() {
        const spec = AppState.wizardInput.target_spec;
        const gridModeRow = document.getElementById('grid_mode_row');
        const toleranceRow = document.getElementById('tolerance-row');

        // Grid Mode is only applicable for:
        // 1. Infinite distance (target_type = 'angle')
        // 2. Finite distance with Strategy 2 (use_strategy2 = true)
        const isFiniteWithoutStrategy2 = spec.target_type === 'size' && !spec.use_strategy2;
        const showGridMode = !isFiniteWithoutStrategy2;

        if (gridModeRow) {
            gridModeRow.style.display = showGridMode ? '' : 'none';
        }

        if (toleranceRow) {
            // Only show tolerance if grid mode is visible AND uniform is selected
            toleranceRow.style.display = showGridMode && spec.grid_mode === 'uniform' ? '' : 'none';
        }
    },

    /**
     * Bind events for target specification form
     */
    bindTargetSpecEvents() {
        const targetType = document.getElementById('target_type');
        const gridMode = document.getElementById('grid_mode');
        const workingDistanceRow = document.getElementById('working-distance-row');
        const toleranceRow = document.getElementById('tolerance-row');
        const targetSpanUnit = document.getElementById('target_span_unit');
        const targetSpanInput = document.getElementById('target_span');

        if (targetType) {
            targetType.addEventListener('change', (e) => {
                const newType = e.target.value;
                const oldType = AppState.wizardInput.target_spec.target_type;

                // Update state
                AppState.wizardInput.target_spec.target_type = newType;

                // Update unit label (for splitters)
                if (targetSpanUnit) {
                    targetSpanUnit.textContent = newType === 'angle' ? '(deg)' : '(mm)';
                }

                // Update diffuser span label (for diffusers)
                // Note: We use a text node approach to avoid losing event bindings on the input
                const diffuserSpanLabel = document.getElementById('diffuser_span_label');
                if (diffuserSpanLabel) {
                    const input = diffuserSpanLabel.querySelector('input');
                    if (input) {
                        // Get the current input value before modifying
                        const currentVal = input.value;
                        // Clear and rebuild label with same input
                        const labelText = newType === 'angle' ? 'Diffusion Angle (deg)' : 'Target Size (mm)';
                        diffuserSpanLabel.innerHTML = '';
                        diffuserSpanLabel.appendChild(document.createTextNode(labelText + ' '));
                        const newInput = document.createElement('input');
                        newInput.type = 'number';
                        newInput.id = 'target_span';
                        newInput.value = currentVal;
                        newInput.step = '0.1';
                        diffuserSpanLabel.appendChild(newInput);
                        // Re-bind the event
                        newInput.addEventListener('change', (e) => {
                            const targetType = AppState.wizardInput.target_spec.target_type;
                            const parsed = this.parseSpanFromDisplay(e.target.value, targetType);
                            if (parsed !== null) {
                                AppState.wizardInput.target_spec.target_span = parsed;
                            }
                        });
                    }
                }

                // Convert current span value to new unit system for display
                // AND update internal state to match the new interpretation
                if (targetSpanInput && oldType !== newType) {
                    const is2D = AppState.wizardInput.doe_type === 'splitter_2d';
                    const currentSpan = AppState.wizardInput.target_spec.target_span;
                    const displayVal = this.formatSpanForDisplay(currentSpan, newType, is2D);
                    targetSpanInput.value = displayVal;
                    // Parse the display value back to internal format with new target_type
                    const newInternalVal = this.parseSpanFromDisplay(displayVal, newType);
                    if (newInternalVal !== null) {
                        AppState.wizardInput.target_spec.target_span = newInternalVal;
                    }
                }

                // Show/hide working distance
                if (workingDistanceRow) {
                    workingDistanceRow.style.display = newType === 'size' ? '' : 'none';
                }

                // Update Grid Mode visibility
                this.updateGridModeVisibility();
            });
        }

        if (gridMode) {
            gridMode.addEventListener('change', (e) => {
                AppState.wizardInput.target_spec.grid_mode = e.target.value;
                if (toleranceRow) {
                    toleranceRow.style.display = e.target.value === 'uniform' ? '' : 'none';
                }
            });
        }

        // Bind other inputs
        this.bindInputIfExists('num_spots', (val) => {
            try {
                const parsed = JSON.parse(val);
                AppState.wizardInput.target_spec.num_spots = parsed;
            } catch {
                AppState.wizardInput.target_spec.num_spots = parseInt(val);
            }
        });

        // Target span: convert from display units (deg/mm) to internal units (rad/m)
        this.bindInputIfExists('target_span', (val) => {
            const targetType = AppState.wizardInput.target_spec.target_type;
            const parsed = this.parseSpanFromDisplay(val, targetType);
            // Only update if parsing succeeded
            if (parsed !== null) {
                AppState.wizardInput.target_spec.target_span = parsed;
            }
        });

        this.bindInputIfExists('tolerance', (val) => {
            AppState.wizardInput.target_spec.tolerance = parseFloat(val) / 100;
        });

        this.bindInputIfExists('working_distance', (val) => {
            AppState.wizardInput.working_distance = val ? parseFloat(val) * 1e-3 : null;
        });

        this.bindInputIfExists('use_strategy2', (val, el) => {
            AppState.wizardInput.target_spec.use_strategy2 = el.checked;
            // Update Grid Mode visibility when Strategy 2 changes
            this.updateGridModeVisibility();
        }, 'checkbox');

        this.bindInputIfExists('focal_length', (val) => {
            // Convert from display mm to internal meters
            AppState.wizardInput.target_spec.focal_length = parseFloat(val) * 1e-3;
        });

        this.bindInputIfExists('lens_type', (val) => {
            AppState.wizardInput.target_spec.lens_type = val;
        });

        this.bindInputIfExists('diffuser_shape', (val) => {
            AppState.wizardInput.target_spec.shape = val;
        });
    },

    /**
     * Bind input event if element exists
     */
    bindInputIfExists(id, handler, type = 'input') {
        const el = document.getElementById(id);
        if (el) {
            const event = type === 'checkbox' ? 'change' : 'change';
            el.addEventListener(event, (e) => handler(e.target.value, e.target));
        }
    },

    /**
     * Collect all wizard input values
     */
    collectInput() {
        // Values are already synced to AppState via event handlers
        return AppState.getWizardRequest();
    },

    /**
     * Generate structured parameters from wizard input
     */
    async generateParams() {
        const btn = document.getElementById('generate-params-btn');
        const startBtn = document.getElementById('start-optimization-btn');
        btn.disabled = true;
        btn.textContent = 'Generating...';

        try {
            const request = this.collectInput();

            // Generate params
            const response = await fetch('/api/wizard', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(request)
            });

            const data = await response.json();

            if (data.success) {
                // Update state from wizard output
                AppState.updateFromWizard(data);

                // Update pattern status to show wizard-generated pattern
                AppState.updatePatternStatus('wizard');

                // Sync UI with updated state
                ParamsUI.render();

                // Also generate preview
                btn.textContent = 'Loading Preview...';
                await PreviewUI.updatePreview();

                // Enable start optimization button
                if (startBtn) {
                    startBtn.disabled = false;
                }

                App.showToast('Parameters generated', 'success');
            } else {
                App.showToast(data.error || 'Failed to generate parameters', 'error');
            }
        } catch (err) {
            console.error('Error generating params:', err);
            App.showToast('Failed to generate parameters', 'error');
        } finally {
            btn.disabled = false;
            btn.textContent = 'Generate Parameters';
        }
    }
};

// Make available globally
window.WizardUI = WizardUI;
