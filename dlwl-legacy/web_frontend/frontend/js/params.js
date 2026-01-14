/**
 * Structured Parameters Panel - Editable parameters decoupled from wizard
 */

const ParamsUI = {
    /**
     * Initialize params panel
     */
    init() {
        this.bindEvents();
        this.syncUIFromState();
    },

    /**
     * Bind event listeners for all editable fields
     */
    bindEvents() {
        // Propagation type
        const propType = document.getElementById('param_prop_type');
        if (propType) {
            propType.addEventListener('change', (e) => {
                AppState.structuredParams.propagation_type = e.target.value;
                this.toggleWorkingDistanceField();

                // Sync to wizardInput for Strategy 2
                if (e.target.value === 'periodic_fresnel') {
                    AppState.wizardInput.target_spec.use_strategy2 = true;
                } else {
                    AppState.wizardInput.target_spec.use_strategy2 = false;
                }
            });
        }

        // Working distance
        const workingDist = document.getElementById('param_working_distance');
        if (workingDist) {
            workingDist.addEventListener('change', (e) => {
                const val = parseFloat(e.target.value);
                const workingDistanceM = isNaN(val) ? null : val * 1e-3;
                AppState.structuredParams.working_distance = workingDistanceM;
                // Sync to wizardInput so it's included in optimization request
                AppState.wizardInput.working_distance = workingDistanceM;
            });
        }

        // Target Span (for ASM/SFR - decoupled from wizard)
        const targetSpan = document.getElementById('param_target_span');
        if (targetSpan) {
            targetSpan.addEventListener('change', (e) => {
                const val = parseFloat(e.target.value);
                const targetSpanM = isNaN(val) ? null : val * 1e-3;  // mm to m
                AppState.structuredParams.target_span = targetSpanM;
                // Sync to wizardInput.target_spec.target_span for physical size mode
                if (targetSpanM !== null) {
                    AppState.wizardInput.target_spec.target_type = 'size';
                    AppState.wizardInput.target_spec.target_span = targetSpanM;
                }
            });
        }

        // DOE pixels
        const doePixels = document.getElementById('param_doe_pixels');
        if (doePixels) {
            doePixels.addEventListener('change', (e) => {
                try {
                    const val = JSON.parse(e.target.value);
                    AppState.structuredParams.doe_pixels = val;
                } catch {
                    const num = parseInt(e.target.value);
                    if (!isNaN(num)) {
                        AppState.structuredParams.doe_pixels = [num, num];
                    }
                }
            });
        }

        // Simulation pixels
        const simPixels = document.getElementById('param_sim_pixels');
        if (simPixels) {
            simPixels.addEventListener('change', (e) => {
                try {
                    const val = JSON.parse(e.target.value);
                    AppState.structuredParams.simulation_pixels = val;
                } catch {
                    const num = parseInt(e.target.value);
                    if (!isNaN(num)) {
                        AppState.structuredParams.simulation_pixels = [num, num];
                    }
                }
            });
        }

        // Advanced settings
        this.bindAdvancedSettings();

        // Optimization settings
        this.bindOptimizationSettings();

        // Validate button
        const validateBtn = document.getElementById('validate-params-btn');
        if (validateBtn) {
            validateBtn.addEventListener('click', () => {
                this.validateAndPreview();
            });
        }
    },

    /**
     * Bind advanced settings events
     */
    bindAdvancedSettings() {
        const maxRes = document.getElementById('adv_max_resolution');
        if (maxRes) {
            maxRes.addEventListener('change', (e) => {
                AppState.advancedSettings.max_resolution = parseInt(e.target.value) || 2000;
            });
        }

        const margin = document.getElementById('adv_target_margin');
        if (margin) {
            margin.addEventListener('change', (e) => {
                AppState.advancedSettings.target_margin = parseInt(e.target.value) || 10;
            });
        }

        const interval = document.getElementById('adv_progress_interval');
        if (interval) {
            interval.addEventListener('change', (e) => {
                AppState.advancedSettings.progress_interval = parseInt(e.target.value) || 50;
            });
        }
    },

    /**
     * Bind optimization settings events
     */
    bindOptimizationSettings() {
        const method = document.getElementById('opt_method');
        if (method) {
            method.addEventListener('change', (e) => {
                AppState.optimizationSettings.phase_method = e.target.value;
            });
        }

        const lr = document.getElementById('opt_lr');
        if (lr) {
            lr.addEventListener('change', (e) => {
                AppState.optimizationSettings.phase_lr = parseFloat(e.target.value) || 3e-9;
            });
        }

        const iters = document.getElementById('opt_iters');
        if (iters) {
            iters.addEventListener('change', (e) => {
                AppState.optimizationSettings.phase_iters = parseInt(e.target.value) || 1000;
            });
        }

        const lossType = document.getElementById('opt_loss_type');
        if (lossType) {
            lossType.addEventListener('change', (e) => {
                AppState.optimizationSettings.loss_type = e.target.value;
            });
        }

        const upsample = document.getElementById('opt_sim_upsample');
        if (upsample) {
            upsample.addEventListener('change', (e) => {
                AppState.optimizationSettings.simulation_upsample = parseInt(e.target.value) || 1;
                // Update reference values to reflect new simulation pixel size
                this.renderReferenceValues();
            });
        }

        const multiplier = document.getElementById('opt_pixel_multiplier');
        if (multiplier) {
            multiplier.addEventListener('change', (e) => {
                AppState.optimizationSettings.pixel_multiplier = parseInt(e.target.value) || 1;
                // Update reference values to reflect new effective pixel size
                this.renderReferenceValues();
            });
        }
    },

    /**
     * Toggle field visibility based on propagation type
     */
    toggleWorkingDistanceField() {
        const wdGroup = document.getElementById('working_distance_group');
        const tsGroup = document.getElementById('target_span_group');
        const tmGroup = document.getElementById('target_margin_group');
        const propType = AppState.structuredParams.propagation_type;

        // Show working distance for all non-FFT propagation types
        const showWD = propType === 'asm' || propType === 'sfr' || propType === 'periodic_fresnel';
        if (wdGroup) {
            wdGroup.style.display = showWD ? '' : 'none';
        }

        // Show target span and target margin only for ASM/SFR (physical space propagation)
        const showASMSFR = propType === 'asm' || propType === 'sfr';
        if (tsGroup) {
            tsGroup.style.display = showASMSFR ? '' : 'none';
        }
        if (tmGroup) {
            tmGroup.style.display = showASMSFR ? '' : 'none';
        }
    },

    /**
     * Sync UI fields from state
     */
    syncUIFromState() {
        const p = AppState.structuredParams;

        this.setInputValue('param_prop_type', p.propagation_type);
        this.setInputValue('param_working_distance', p.working_distance ? p.working_distance * 1e3 : '');
        this.setInputValue('param_target_span', p.target_span ? p.target_span * 1e3 : '');
        this.setInputValue('param_doe_pixels', JSON.stringify(p.doe_pixels));
        this.setInputValue('param_sim_pixels', JSON.stringify(p.simulation_pixels));

        // Advanced settings
        this.setInputValue('adv_max_resolution', AppState.advancedSettings.max_resolution);
        this.setInputValue('adv_target_margin', AppState.advancedSettings.target_margin);
        this.setInputValue('adv_progress_interval', AppState.advancedSettings.progress_interval);

        // Optimization settings
        this.setInputValue('opt_method', AppState.optimizationSettings.phase_method);
        this.setInputValue('opt_lr', AppState.optimizationSettings.phase_lr);
        this.setInputValue('opt_iters', AppState.optimizationSettings.phase_iters);
        this.setInputValue('opt_loss_type', AppState.optimizationSettings.loss_type);
        this.setInputValue('opt_sim_upsample', AppState.optimizationSettings.simulation_upsample);
        this.setInputValue('opt_pixel_multiplier', AppState.optimizationSettings.pixel_multiplier);

        this.toggleWorkingDistanceField();
    },

    /**
     * Set input value if element exists
     */
    setInputValue(id, value) {
        const el = document.getElementById(id);
        if (el) {
            el.value = value;
        }
    },

    /**
     * Render reference values (read-only section)
     * All values come from DOE Settings (structuredParams), NOT wizard
     */
    renderReferenceValues() {
        const container = document.getElementById('reference-values');
        if (!container) return;

        const ref = AppState.referenceValues;
        const computed = AppState.computedValues || {};
        const params = AppState.structuredParams;
        const optSettings = AppState.optimizationSettings;

        const items = [];

        // All physical values come from DOE Settings (structuredParams)
        const wavelength = params.wavelength;
        const pixelSize = params.pixel_size;
        // Device diameter = doe_pixels × pixel_size
        const doePixels = Array.isArray(params.doe_pixels) ? params.doe_pixels[0] : params.doe_pixels;
        const doeDiameter = doePixels * pixelSize;
        const pixelMultiplier = optSettings.pixel_multiplier || 1;
        const simUpsample = optSettings.simulation_upsample || 1;

        // Period in meters
        // For ASM/SFR (non-periodic), show "Full Device"
        const propType = params.propagation_type;
        const isNonPeriodic = propType === 'asm' || propType === 'sfr';

        if (isNonPeriodic) {
            items.push({ label: 'Period', value: 'Full Device', title: 'Non-periodic DOE (ASM/SFR propagation)' });
        } else if (ref.period_meters !== undefined && ref.period_meters !== null) {
            const periodUm = ref.period_meters * 1e6;
            items.push({ label: 'Period', value: `${periodUm.toFixed(2)} um` });
        }

        // Note: Num Periods is not shown to users (auto-tiled by default)

        // DOE pixel size - shows effective pixel (pixel_size × pixel_multiplier)
        // This is the actual pixel size used in optimization and determines max diffraction angle
        if (pixelSize) {
            const effectivePixelSize = pixelSize * pixelMultiplier;
            const effectivePixelUm = effectivePixelSize * 1e6;
            if (pixelMultiplier > 1) {
                items.push({
                    label: 'DOE Pixel',
                    value: `${effectivePixelUm.toFixed(2)} um (${pixelMultiplier}×)`,
                    title: `Effective DOE pixel = ${(pixelSize * 1e6).toFixed(2)} um × ${pixelMultiplier} (pixel multiplier)`
                });
            } else {
                items.push({
                    label: 'DOE Pixel',
                    value: `${effectivePixelUm.toFixed(2)} um`,
                    title: 'DOE pixel size (determines max diffraction angle)'
                });
            }
        }

        // Simulation pixel (with Sim Upsample) - always show
        // simulation pixel = effective DOE pixel / simUpsample
        if (pixelSize) {
            const effectivePixelSize = pixelSize * pixelMultiplier;
            const simPixelUm = effectivePixelSize / simUpsample * 1e6;
            const suffix = simUpsample > 1 ? ` (${simUpsample}× upsample)` : '';
            items.push({ label: 'Sim Pixel', value: `${simPixelUm.toFixed(3)} um`, title: `Simulation pixel size during optimization${suffix}` });
        }

        // Max diffraction angle (limited by effective pixel size)
        // sin(θ_max) = λ / (2 × pixel_effective)
        if (wavelength && pixelSize) {
            const effectivePixel = pixelSize * pixelMultiplier;
            const sinTheta = wavelength / (2 * effectivePixel);
            if (sinTheta <= 1) {
                const maxAngleRad = Math.asin(sinTheta);
                const maxAngleDeg = maxAngleRad * 180 / Math.PI;
                items.push({ label: 'Max Angle', value: `${maxAngleDeg.toFixed(2)} deg`, title: `Max diffraction angle (λ/2p, mult=${pixelMultiplier}×)` });
            } else {
                items.push({ label: 'Max Angle', value: '> 90 deg', title: 'Pixel size smaller than λ/2' });
            }
        }

        // Current pattern's max angle (diagonal)
        if (computed.order_angles && computed.order_angles.length > 0) {
            let maxPatternAngle = 0;
            computed.order_angles.forEach(angle => {
                if (Array.isArray(angle)) {
                    const diagAngle = Math.sqrt(angle[0] * angle[0] + angle[1] * angle[1]);
                    maxPatternAngle = Math.max(maxPatternAngle, diagAngle);
                } else {
                    maxPatternAngle = Math.max(maxPatternAngle, Math.abs(angle));
                }
            });
            const patternAngleDeg = maxPatternAngle * 180 / Math.PI;
            items.push({ label: 'Pattern Max', value: `${patternAngleDeg.toFixed(2)} deg`, title: 'Maximum angle in current pattern (diagonal)' });
        }

        // Diffraction limit (angular resolution) - λ/D in degrees
        if (wavelength && doeDiameter) {
            const diffLimitRad = wavelength / doeDiameter;
            const diffLimitDeg = diffLimitRad * 180 / Math.PI;
            // Show in deg (not mdeg as per user request)
            items.push({ label: 'Diff. Limit', value: `${diffLimitDeg.toFixed(4)} deg`, title: 'Angular resolution λ/D' });
        }

        // Min Tolerance (when period = full device diameter, i.e., simulation_pixels = doe_pixels)
        // This represents the worst-case tolerance when only 1 period fits in the device
        // Backend formula: tolerance = k_step / (2 × target_span)
        // where k_step = λ/period and target_span = 2 × max_angle (full span from -max to +max)
        // So: tolerance = (λ/period) / (4 × max_angle)
        // When period = D (device diameter): min_tolerance = (λ/D) / (4 × max_angle)
        if (wavelength && doeDiameter && computed.order_angles && computed.order_angles.length > 0) {
            // K-space step when period = device diameter
            const kStep = wavelength / doeDiameter;  // radians
            // Pattern max angle (per axis, not diagonal)
            let maxPatternAnglePerAxis = 0;
            computed.order_angles.forEach(angle => {
                if (Array.isArray(angle)) {
                    maxPatternAnglePerAxis = Math.max(maxPatternAnglePerAxis, Math.abs(angle[0]), Math.abs(angle[1]));
                } else {
                    maxPatternAnglePerAxis = Math.max(maxPatternAnglePerAxis, Math.abs(angle));
                }
            });
            if (maxPatternAnglePerAxis > 0) {
                // target_span = 2 × max_angle, tolerance = k_step / (2 × target_span)
                const targetSpan = 2 * maxPatternAnglePerAxis;
                const minTolerance = kStep / (2 * targetSpan);
                items.push({ label: 'Min Tolerance', value: `${(minTolerance * 100).toFixed(1)}%`, title: 'Minimum tolerance when simulation_pixels = doe_pixels (1 period)' });
            }
        }

        // Max pixel multiplier - calculate from Pattern Max
        // Max multiplier such that max_angle(effective_pixel) >= pattern_max
        // sin(pattern_max) <= λ / (2 × pixel × multiplier)
        // multiplier <= λ / (2 × pixel × sin(pattern_max))
        if (wavelength && pixelSize && computed.order_angles && computed.order_angles.length > 0) {
            let maxPatternAngle = 0;
            computed.order_angles.forEach(angle => {
                if (Array.isArray(angle)) {
                    // Use per-axis max (not diagonal) for safety
                    maxPatternAngle = Math.max(maxPatternAngle, Math.abs(angle[0]), Math.abs(angle[1]));
                } else {
                    maxPatternAngle = Math.max(maxPatternAngle, Math.abs(angle));
                }
            });
            if (maxPatternAngle > 0) {
                const sinPatternMax = Math.sin(maxPatternAngle);
                const maxMult = Math.floor(wavelength / (2 * pixelSize * sinPatternMax));
                const displayMult = Math.max(1, Math.min(maxMult, 10));  // Clamp 1-10
                items.push({ label: 'Max Mult.', value: `${displayMult}`, title: `Max pixel multiplier to achieve pattern (based on Pattern Max = ${(maxPatternAngle * 180 / Math.PI).toFixed(2)}°)` });
            }
        } else if (ref.max_pixel_multiplier !== undefined) {
            items.push({ label: 'Max Mult.', value: `${ref.max_pixel_multiplier}`, title: 'Max pixel multiplier for current pattern' });
        }

        // Number of orders/spots - use num_orders (non-zero pixels in target) if available
        // This comes from computed values (generated during validation), not wizard directly
        if (computed.num_orders !== undefined) {
            items.push({ label: '# Orders', value: computed.num_orders, title: 'Number of non-zero pixels in target pattern' });
        } else if (computed.working_orders) {
            items.push({ label: '# Orders', value: computed.working_orders.length, title: 'Number of diffraction orders' });
        } else if (computed.array_size) {
            // For lens array, show from computed values
            const [ny, nx] = Array.isArray(computed.array_size) ? computed.array_size : [1, 1];
            items.push({ label: '# Foci', value: `${ny} × ${nx}`, title: 'Number of focal spots' });
        }

        if (items.length === 0) {
            container.innerHTML = '<p class="hint">Generate parameters to see computed values</p>';
            return;
        }

        container.innerHTML = items.map(item =>
            `<div class="reference-item" ${item.title ? `title="${item.title}"` : ''}>
                <span class="ref-label">${item.label}</span>
                <span class="ref-value">${item.value}</span>
            </div>`
        ).join('');
    },

    /**
     * Render warnings
     */
    renderWarnings() {
        const section = document.getElementById('warnings-section');
        const container = document.getElementById('validation-warnings');

        if (!section || !container) return;

        const warnings = AppState.warnings;

        if (!warnings || warnings.length === 0) {
            section.style.display = 'none';
            return;
        }

        section.style.display = 'block';
        container.innerHTML = warnings.map(w =>
            `<div class="warning-item">
                <strong>${w.code || 'Warning'}:</strong> ${w.message}
            </div>`
        ).join('');
    },

    /**
     * Full render of params panel
     */
    render() {
        this.syncUIFromState();
        this.renderReferenceValues();
        this.renderWarnings();
    },

    /**
     * Validate parameters and update preview
     */
    async validateAndPreview() {
        const btn = document.getElementById('validate-params-btn');
        const startBtn = document.getElementById('start-optimization-btn');

        btn.disabled = true;
        btn.textContent = 'Validating...';

        try {
            // Use wizard request for validation (for now)
            const validateResponse = await fetch('/api/validate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(AppState.getWizardRequest())
            });

            const validateData = await validateResponse.json();

            if (!validateData.is_valid) {
                const errorHtml = validateData.errors.map(e =>
                    `<div class="error-item">
                        <strong>${e.code}:</strong> ${e.message}
                    </div>`
                ).join('');

                document.getElementById('warnings-section').style.display = 'block';
                document.getElementById('validation-warnings').innerHTML = errorHtml;

                App.showToast('Validation failed', 'error');
                return;
            }

            // Update warnings
            if (validateData.warnings) {
                AppState.warnings = validateData.warnings;
                this.renderWarnings();
            }

            // Generate preview
            btn.textContent = 'Loading Preview...';
            await PreviewUI.updatePreview();

            // Update reference values (reflects current pixel multiplier, sim upsample, etc.)
            this.renderReferenceValues();

            // Enable start button
            if (startBtn) {
                startBtn.disabled = false;
            }

            App.showToast('Parameters validated', 'success');
        } catch (err) {
            console.error('Validation error:', err);
            App.showToast('Validation failed', 'error');
        } finally {
            btn.disabled = false;
            btn.textContent = 'Validate & Update Preview';
        }
    }
};

// Make available globally
window.ParamsUI = ParamsUI;
