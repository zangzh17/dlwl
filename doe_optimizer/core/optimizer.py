"""
Optimization algorithms for DOE design.

Supports three algorithms:
1. SGD - Stochastic Gradient Descent with Adam optimizer
2. GS - Gerchberg-Saxton iterative algorithm
3. BS - Binary Search (for 1D or small 2D problems)
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .propagation import get_propagator
from ..utils.image_utils import crop_pad_image, create_circular_mask
from ..utils.math_utils import compute_loss, height2phase


class Optimizer(ABC):
    """Base class for DOE optimization algorithms."""

    def __init__(
        self,
        propagator: Callable,
        prop_model: str,
        feature_size: tuple,
        wavelength: np.ndarray,
        refraction_index: np.ndarray,
        prop_dist: np.ndarray,
        roi_resolution: tuple,
        output_resolution: tuple = None,
        output_size: tuple = None,
        aperture_type: str = 'square',
        fab_model: nn.Module = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32
    ):
        """Initialize optimizer.

        Args:
            propagator: Propagation function
            prop_model: Propagation model name ('ASM', 'FFT', 'SFR', 'None')
            feature_size: (dy, dx) pixel size
            wavelength: Wavelength array [1, C, 1, 1]
            refraction_index: Refractive index array [1, C, 1, 1]
            prop_dist: Propagation distance array [1, C, 1, 1]
            roi_resolution: ROI resolution for loss calculation
            output_resolution: Output resolution after propagation
            output_size: Physical output size (for SFR)
            aperture_type: 'square' or 'circular'
            fab_model: Fabrication model (optional)
            device: Torch device
            dtype: Torch dtype
        """
        self.propagator = propagator
        self.prop_model = prop_model
        self.feature_size = feature_size
        self.wavelength = wavelength
        self.refraction_index = refraction_index
        self.prop_dist = prop_dist
        self.roi_resolution = roi_resolution
        self.output_resolution = output_resolution
        self.output_size = output_size
        self.aperture_type = aperture_type
        self.fab_model = fab_model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype

        # Pre-computed kernels
        self.precomputed_H = None
        self.zfft2 = None

    @abstractmethod
    def optimize(
        self,
        target: torch.Tensor,
        init_value: torch.Tensor,
        num_iters: int,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Run optimization.

        Args:
            target: Target amplitude [B, C, H, W]
            init_value: Initial guess [B, 1, H, W]
            num_iters: Number of iterations
            **kwargs: Algorithm-specific parameters

        Returns:
            Tuple of (optimized_value, reconstructed_amplitude, final_loss)
        """
        pass

    def forward_model(
        self,
        value: torch.Tensor,
        is_dose: bool = False
    ) -> torch.Tensor:
        """Apply forward model: value -> reconstructed field.

        Args:
            value: Height [B, 1, H, W] or dose if is_dose=True
            is_dose: If True, apply fabrication model first

        Returns:
            Complex field [B, C, H, W]
        """
        if is_dose and self.fab_model is not None:
            height = self.fab_model(value, backward=False)
        else:
            height = value

        # Apply aperture mask
        if self.aperture_type == 'circular':
            mask = create_circular_mask(height.shape[-2:], device=self.device)
            height = height * mask

        # Convert height to complex field
        phase = height2phase(height, self.wavelength, self.refraction_index, self.dtype)
        field = torch.exp(1j * phase.to(torch.complex64))

        # Propagate
        if self.propagator is not None and self.prop_model != 'None':
            if self.prop_model == 'ASM':
                field = self.propagator(
                    field, self.feature_size, self.wavelength, self.prop_dist,
                    output_resolution=self.output_resolution,
                    precomputed_H=self.precomputed_H,
                    dtype=self.dtype
                )
            elif self.prop_model == 'FFT':
                field = self.propagator(
                    field, output_resolution=self.output_resolution, z=1
                )
            elif self.prop_model == 'SFR':
                field = self.propagator(
                    field, self.feature_size, self.wavelength, self.prop_dist,
                    output_size=self.output_size, output_resolution=self.output_resolution,
                    zfft2=self.zfft2, precomputed_H=self.precomputed_H,
                    dtype=self.dtype
                )

        return field

    def _precompute_kernels(self, shape: tuple):
        """Pre-compute propagation kernels for efficiency."""
        if self.propagator is None or self.prop_model == 'None':
            return

        dummy = torch.empty(*shape, dtype=torch.complex64, device=self.device)

        if self.prop_model == 'ASM':
            self.precomputed_H = self.propagator(
                dummy, self.feature_size, self.wavelength, self.prop_dist,
                output_resolution=self.output_resolution,
                return_H=True, dtype=self.dtype
            )
            if self.precomputed_H is not None:
                self.precomputed_H = self.precomputed_H.to(self.device).detach()
                self.precomputed_H.requires_grad = False

        elif self.prop_model == 'SFR':
            self.zfft2 = self.propagator(
                dummy, self.feature_size, self.wavelength, self.prop_dist,
                output_size=self.output_size, output_resolution=self.output_resolution,
                return_zfft2=True, dtype=self.dtype
            )
            self.precomputed_H = self.propagator(
                dummy, self.feature_size, self.wavelength, self.prop_dist,
                output_size=self.output_size, output_resolution=self.output_resolution,
                return_H=True, zfft2=self.zfft2, dtype=self.dtype
            )
            if isinstance(self.precomputed_H, tuple):
                self.precomputed_H = tuple(
                    h.to(self.device).detach() for h in self.precomputed_H
                )


class SGDOptimizer(Optimizer):
    """Stochastic Gradient Descent optimizer with Adam."""

    def optimize(
        self,
        target: torch.Tensor,
        init_value: torch.Tensor,
        num_iters: int,
        lr: float = 0.3,
        min_value: float = 0.0,
        max_value: float = None,
        loss_type: str = 'L2',
        optimizer_type: str = 'adam',
        upsample_factor: int = 1,
        progress_callback: Callable = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Run SGD optimization.

        Args:
            target: Target amplitude [B, C, H, W]
            init_value: Initial guess [B, 1, H, W]
            num_iters: Number of iterations
            lr: Learning rate
            min_value: Minimum value constraint
            max_value: Maximum value constraint
            loss_type: Loss function type
            optimizer_type: 'adam' or 'sgd'
            upsample_factor: Upsampling factor for phase simulation
            progress_callback: Optional callback(iter, loss) for progress

        Returns:
            Tuple of (optimized_value, reconstructed_amplitude, final_loss)
        """
        from ..utils.image_utils import upsample_nearest

        # For upsampling, we need to adjust the shape for kernel pre-computation
        upsampled_shape = list(init_value.shape)
        if upsample_factor > 1:
            upsampled_shape[-2] *= upsample_factor
            upsampled_shape[-1] *= upsample_factor

        # Pre-compute kernels with upsampled shape
        self._precompute_kernels(tuple(upsampled_shape))

        # Resize target to ROI resolution (adjusted for upsampling)
        if upsample_factor > 1:
            target_roi = (self.roi_resolution[0] * upsample_factor,
                          self.roi_resolution[1] * upsample_factor)
        else:
            target_roi = self.roi_resolution
        target = crop_pad_image(target, target_roi).to(self.device)

        # Initialize optimization variable
        x = init_value.clone().detach().requires_grad_(True)
        min_val = torch.tensor(min_value, device=self.device)
        max_val = torch.tensor(max_value, device=self.device) if max_value else None

        # Create optimizer
        if optimizer_type == 'adam':
            optimizer = optim.Adam([x], lr=lr, eps=1e-8)
        else:
            optimizer = optim.SGD([x], lr=lr, momentum=0.9)

        # Optimization loop
        for k in range(num_iters):
            optimizer.zero_grad()

            # Clamp values
            with torch.no_grad():
                x.data = torch.clamp(x.data, min_val, max_val) if max_val else torch.clamp(x.data, min_val)

            # Upsample phase before forward model if needed
            if upsample_factor > 1:
                x_upsampled = upsample_nearest(x, upsample_factor)
            else:
                x_upsampled = x

            # Forward model
            is_dose = self.fab_model is not None and self.prop_model != 'None'
            field = self.forward_model(x_upsampled, is_dose=is_dose)
            recon_amp = field.abs()

            # Crop to ROI for loss computation (matching original code)
            recon_amp_roi = crop_pad_image(recon_amp, target_roi)

            # Compute loss
            loss = compute_loss(recon_amp_roi, target, loss_type)
            loss.backward()
            optimizer.step()

            # Report progress every 100 iterations or at specific points
            if progress_callback and (k % 100 == 0 or k == num_iters - 1):
                progress_callback(k, float(loss.item()))

        # Final forward pass
        with torch.no_grad():
            x.data = torch.clamp(x.data, min_val, max_val) if max_val else torch.clamp(x.data, min_val)

            if upsample_factor > 1:
                x_upsampled = upsample_nearest(x, upsample_factor)
            else:
                x_upsampled = x

            field = self.forward_model(x_upsampled, is_dose=is_dose)
            recon_amp = field.abs()
            recon_amp_roi = crop_pad_image(recon_amp, target_roi)
            final_loss = float(compute_loss(recon_amp_roi, target, loss_type).item())

        return x.detach(), recon_amp.detach(), final_loss


class GSOptimizer(Optimizer):
    """Gerchberg-Saxton iterative optimizer."""

    def optimize(
        self,
        target: torch.Tensor,
        init_value: torch.Tensor,
        num_iters: int,
        progress_callback: Callable = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Run GS optimization.

        Args:
            target: Target amplitude [B, C, H, W]
            init_value: Initial guess [B, 1, H, W]
            num_iters: Number of iterations
            progress_callback: Optional callback(iter, loss) for progress

        Returns:
            Tuple of (optimized_value, reconstructed_amplitude, final_loss)
        """
        self._precompute_kernels(init_value.shape)

        target = crop_pad_image(target, self.roi_resolution).to(self.device)
        x = init_value.clone()

        roi_mask = create_roi_mask(
            self.output_resolution or init_value.shape[-2:],
            self.roi_resolution,
            device=self.device
        )

        for k in range(num_iters):
            # Forward: height -> field
            is_dose = self.fab_model is not None and self.prop_model != 'None'
            if is_dose:
                height = self.fab_model(x, backward=False)
            else:
                height = x

            phase = height2phase(height, self.wavelength, self.refraction_index, self.dtype)
            field = torch.exp(1j * phase.to(torch.complex64))

            # Propagate forward
            if self.propagator and self.prop_model != 'None':
                if self.prop_model == 'FFT':
                    recon_field = self.propagator(field, output_resolution=self.output_resolution)
                elif self.prop_model == 'ASM':
                    recon_field = self.propagator(
                        field, self.feature_size, self.wavelength, self.prop_dist,
                        output_resolution=self.output_resolution,
                        precomputed_H=self.precomputed_H
                    )
                else:
                    recon_field = field
            else:
                recon_field = field

            # Replace amplitude at target plane
            recon_field_masked = recon_field.clone()
            mask = roi_mask.bool().expand_as(recon_field)
            recon_field_masked[mask] = torch.exp(1j * recon_field[mask].angle()) * target[roi_mask.bool().expand_as(target)]

            # Propagate backward
            if self.propagator and self.prop_model != 'None':
                if self.prop_model == 'FFT':
                    back_field = self.propagator(recon_field_masked, z=-1)
                elif self.prop_model == 'ASM':
                    back_field = self.propagator(
                        recon_field_masked, self.feature_size, self.wavelength, -self.prop_dist,
                        precomputed_H=None
                    )
                else:
                    back_field = recon_field_masked
            else:
                back_field = recon_field_masked

            # Extract phase and convert back
            new_phase = back_field.angle()
            new_height = new_phase * self.wavelength[0, 0, 0, 0] / (2 * np.pi * (self.refraction_index[0, 0, 0, 0] - 1))

            if is_dose:
                x = self.fab_model(new_height, backward=True)
            else:
                x = new_height

            if progress_callback and k % 100 == 0:
                loss = compute_loss(recon_field.abs(), target, 'L2', roi_mask)
                progress_callback(k, float(loss.item()))

        # Final evaluation
        final_field = self.forward_model(x, is_dose=is_dose)
        final_loss = float(compute_loss(final_field.abs(), target, 'L2', roi_mask).item())

        return x.detach(), final_field.abs().detach(), final_loss


class BSOptimizer(Optimizer):
    """Binary Search optimizer for 1D or small 2D problems."""

    def optimize(
        self,
        target: torch.Tensor,
        init_value: torch.Tensor,
        num_iters: int,
        min_value: float = 0.0,
        max_value: float = 255.0,
        levels: int = 255,
        progress_callback: Callable = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, float]:
        """Run binary search optimization.

        Args:
            target: Target amplitude [B, C, H, W]
            init_value: Initial guess [B, 1, H, W]
            num_iters: Number of iterations (full passes)
            min_value: Minimum value
            max_value: Maximum value
            levels: Number of quantization levels
            progress_callback: Optional callback(iter, loss) for progress

        Returns:
            Tuple of (optimized_value, reconstructed_amplitude, final_loss)
        """
        self._precompute_kernels(init_value.shape)

        target = crop_pad_image(target, self.roi_resolution).to(self.device)
        x = init_value.clone()

        roi_mask = create_roi_mask(
            self.output_resolution or init_value.shape[-2:],
            self.roi_resolution,
            device=self.device
        )

        delta = (max_value - min_value) / levels
        is_dose = self.fab_model is not None and self.prop_model != 'None'

        # Initial loss
        field = self.forward_model(x, is_dose=is_dose)
        best_loss = float(compute_loss(field.abs(), target, 'L2', roi_mask).item())

        for k in range(num_iters):
            # Random permutation of pixels
            flat_x = x.view(-1)
            rand_idx = torch.randperm(len(flat_x))

            for i in rand_idx:
                # Try +delta
                flat_x[i] = torch.clamp(flat_x[i] + delta, min_value, max_value)
                field = self.forward_model(x.view(init_value.shape), is_dose=is_dose)
                loss_plus = float(compute_loss(field.abs(), target, 'L2', roi_mask).item())

                if loss_plus < best_loss:
                    best_loss = loss_plus
                else:
                    # Try -delta
                    flat_x[i] = torch.clamp(flat_x[i] - 2 * delta, min_value, max_value)
                    field = self.forward_model(x.view(init_value.shape), is_dose=is_dose)
                    loss_minus = float(compute_loss(field.abs(), target, 'L2', roi_mask).item())

                    if loss_minus < best_loss:
                        best_loss = loss_minus
                    else:
                        # Revert
                        flat_x[i] = flat_x[i] + delta

            if progress_callback:
                progress_callback(k, best_loss)

        # Final evaluation
        x = x.view(init_value.shape)
        field = self.forward_model(x, is_dose=is_dose)
        final_loss = float(compute_loss(field.abs(), target, 'L2', roi_mask).item())

        return x.detach(), field.abs().detach(), final_loss


def create_optimizer(
    method: str,
    config: 'DOEConfig',
    fab_model: nn.Module = None,
    for_phase: bool = True,
    device: torch.device = None
) -> Optimizer:
    """Create optimizer instance from configuration.

    Args:
        method: 'SGD', 'GS', or 'BS'
        config: DOEConfig instance
        fab_model: Fabrication model (for fab optimization)
        for_phase: If True, create optimizer for phase optimization
        device: Torch device

    Returns:
        Optimizer instance
    """
    prop_model_name = config.prop_model.value if for_phase else 'None'
    propagator = get_propagator(prop_model_name)

    wavelength = config.physical.wavelength_array
    refraction_index = config.physical.refraction_index_array
    prop_dist = config.physical.working_distance_array if config.physical.working_distance else np.array([[[[1.0]]]])

    # For splitters, use splitter-specific resolution (small, matching diffraction orders)
    # Exception: ASM strategy (Strategy 1) uses full device resolution
    from ..core.config import FiniteDistanceStrategy
    strategy = config.get_finite_distance_strategy() if config.is_splitter() else None

    if config.is_splitter() and for_phase and strategy != FiniteDistanceStrategy.ASM:
        # Periodic optimization (infinite distance or Strategy 2)
        splitter_res = config.get_splitter_resolution()
        roi_res = splitter_res
        output_res = splitter_res
        # Feature size scaled so that N * dx = period
        # This ensures FFT pixel k maps to diffraction order m = k - N/2
        # Since sin(θ_m) = m * λ / period and FFT gives sin(θ) = λ * k / (N * dx)
        period = config.get_splitter_period()
        pixel_size_scaled = period / splitter_res[0]
        feature_size = (pixel_size_scaled, pixel_size_scaled)
    else:
        feature_size = config.get_feature_size(for_phase=for_phase)
        roi_res = config.target.roi_resolution or config.phase_resolution if for_phase else config.slm_resolution
        output_res = config.phase_resolution if for_phase else config.slm_resolution

    common_args = dict(
        propagator=propagator,
        prop_model=prop_model_name,
        feature_size=feature_size,
        wavelength=wavelength,
        refraction_index=refraction_index,
        prop_dist=prop_dist,
        roi_resolution=roi_res,
        output_resolution=output_res,
        output_size=config.target.output_size,
        aperture_type=config.device.shape,
        fab_model=fab_model if not for_phase else None,
        device=device,
    )

    if method == 'SGD':
        return SGDOptimizer(**common_args)
    elif method == 'GS':
        return GSOptimizer(**common_args)
    elif method == 'BS':
        return BSOptimizer(**common_args)
    else:
        raise ValueError(f"Unknown optimization method: {method}")
