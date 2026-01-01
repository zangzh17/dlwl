"""
Fabrication model for laser direct writing (LDW/DLW) process simulation.

The model includes:
1. GT curve: Dose-to-thickness nonlinear mapping
2. LP curve: MTF/PSF for optical proximity effect (OPE)

Reference: Calibration data from MATLAB .mat files
"""

import numpy as np
import torch
import torch.nn as nn


class FitModel:
    """Calibration curve fitting model.

    Loads and processes GT (grayscale-to-thickness) and LP (low-pass/MTF) curves
    from calibration data.
    """

    def __init__(
        self,
        gt_data: dict,
        lp_data: dict,
        lp_ratio: float = 1.0,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32
    ):
        """Initialize fit model from calibration data.

        Args:
            gt_data: GT curve data dictionary (from .mat file)
                Expected keys: 'gt_model' containing dose/gt coefficients
            lp_data: LP curve data dictionary (from .mat file)
                Expected keys: 'lp_model' containing MTF coefficients
            lp_ratio: Scaling factor for LP curve
            device: Torch device
            dtype: Torch dtype
        """
        self.device = device or torch.device('cpu')
        self.dtype = dtype
        self.lp_ratio = lp_ratio

        # Parse GT data
        self._parse_gt_data(gt_data)
        # Parse LP data
        self._parse_lp_data(lp_data)

    def _parse_gt_data(self, gt_data: dict):
        """Parse GT curve data from .mat file format."""
        # Handle nested MATLAB structure
        if 'gt_model' in gt_data:
            gt = gt_data['gt_model']
            # MATLAB struct format: gt['field'][0][0][0]
            self.gt_x = torch.tensor(
                gt['dose_x'][0][0][0], device=self.device, dtype=self.dtype
            )
            self.gt_y = torch.tensor(
                gt['dose_y'][0][0][0], device=self.device, dtype=self.dtype
            )
            self.gt_coeff = torch.tensor(
                gt['dose_coeff'][0][0][0], device=self.device, dtype=self.dtype
            )
            self.gt_inv_x = torch.tensor(
                gt['gt_x'][0][0][0], device=self.device, dtype=self.dtype
            )
            self.gt_inv_y = torch.tensor(
                gt['gt_y'][0][0][0], device=self.device, dtype=self.dtype
            )
            self.gt_inv_coeff = torch.tensor(
                gt['gt_coeff'][0][0][0], device=self.device, dtype=self.dtype
            )
        else:
            # Simplified format
            self.gt_x = torch.tensor(gt_data.get('gt_x', [0, 255]), device=self.device, dtype=self.dtype)
            self.gt_y = torch.tensor(gt_data.get('gt_y', [0, 1]), device=self.device, dtype=self.dtype)
            self.gt_coeff = torch.tensor(gt_data.get('gt_coeff', [0, 1]), device=self.device, dtype=self.dtype)
            self.gt_inv_x = self.gt_y.clone()
            self.gt_inv_y = self.gt_x.clone()
            self.gt_inv_coeff = torch.tensor([1, 0], device=self.device, dtype=self.dtype)

        # Set requires_grad to False for calibration data
        for attr in ['gt_x', 'gt_y', 'gt_coeff', 'gt_inv_x', 'gt_inv_y', 'gt_inv_coeff']:
            getattr(self, attr).requires_grad = False

        self.gt_mid = float((self.gt_x[0] + self.gt_x[1]) / 2)

    def _parse_lp_data(self, lp_data: dict):
        """Parse LP curve data from .mat file format."""
        if 'lp_model' in lp_data:
            lp = lp_data['lp_model']
            self.lp_f0 = torch.tensor(
                lp['f_max'][0][0][0], device=self.device, dtype=self.dtype
            )
            self.lp_coeff = torch.tensor(
                lp['mtf_coeff'][0][0][0], device=self.device, dtype=self.dtype
            )
            self.lp_f0_np = float(lp['f_max'][0][0][0])
        else:
            # Default: flat MTF
            self.lp_f0 = torch.tensor([1.0], device=self.device, dtype=self.dtype)
            self.lp_coeff = torch.tensor([1.0], device=self.device, dtype=self.dtype)
            self.lp_f0_np = 1.0

        self.lp_f0.requires_grad = False
        self.lp_coeff.requires_grad = False

    def gt_torch(self, dose: torch.Tensor) -> torch.Tensor:
        """Apply GT curve: dose -> depth (in um).

        Uses piecewise polynomial interpolation.

        Args:
            dose: Dose values [0-255]

        Returns:
            Depth values in micrometers
        """
        # Find segment index
        x_vals = self.gt_x
        coeff = self.gt_coeff

        # Linear interpolation for simplicity
        # For piecewise polynomial, implement proper segment selection
        dose_clamped = torch.clamp(dose, x_vals[0], x_vals[-1])

        # Simple polynomial evaluation (assuming single segment)
        if len(coeff) >= 2:
            depth = coeff[0] + coeff[1] * dose_clamped
            if len(coeff) >= 3:
                depth = depth + coeff[2] * dose_clamped**2
            if len(coeff) >= 4:
                depth = depth + coeff[3] * dose_clamped**3
        else:
            depth = dose_clamped * (self.gt_y[-1] - self.gt_y[0]) / (x_vals[-1] - x_vals[0])

        return depth

    def gt_inv_torch(self, depth: torch.Tensor) -> torch.Tensor:
        """Apply inverse GT curve: depth (um) -> dose.

        Args:
            depth: Depth values in micrometers

        Returns:
            Dose values [0-255]
        """
        x_vals = self.gt_inv_x
        coeff = self.gt_inv_coeff

        depth_clamped = torch.clamp(depth, x_vals[0], x_vals[-1])

        if len(coeff) >= 2:
            dose = coeff[0] + coeff[1] * depth_clamped
            if len(coeff) >= 3:
                dose = dose + coeff[2] * depth_clamped**2
            if len(coeff) >= 4:
                dose = dose + coeff[3] * depth_clamped**3
        else:
            dose = depth_clamped * 255.0 / (x_vals[-1] - x_vals[0])

        return dose

    def lp(self, freq: np.ndarray) -> np.ndarray:
        """Apply LP (MTF) curve.

        Args:
            freq: Spatial frequency in um^-1

        Returns:
            MTF values
        """
        # Polynomial MTF model
        f_norm = freq / self.lp_f0_np
        coeff = self.lp_coeff.cpu().numpy()

        mtf = np.ones_like(freq)
        for i, c in enumerate(coeff):
            mtf = mtf + c * f_norm**(i + 1) if i > 0 else c * np.ones_like(freq)

        # Apply ratio scaling
        mtf = 1 - (1 - mtf) * self.lp_ratio

        return np.clip(mtf, 0.0, 1.0)


class FabricationModel(nn.Module):
    """Fabrication process model for DOE optimization.

    Models the laser direct writing (LDW) process including:
    - PSF/MTF low-pass filtering (optical proximity effect)
    - Grayscale-to-thickness nonlinear mapping

    Forward: dose -> height
    Backward: height -> dose (for initialization)
    """

    def __init__(
        self,
        fit_model: FitModel,
        feature_size: tuple,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32
    ):
        """Initialize fabrication model.

        Args:
            fit_model: FitModel instance with calibration data
            feature_size: (dy, dx) pixel size in meters
            device: Torch device
            dtype: Torch dtype
        """
        super().__init__()

        self.model_gt = fit_model.gt_torch
        self.model_gt_inv = fit_model.gt_inv_torch
        self.model_lp = fit_model.lp
        self.device = device or torch.device('cpu')
        self.dtype = dtype

        self.feature_size = feature_size
        self.precomputed_H = None
        self.precomputed_H_b = None
        self.field_resolution = None

        # Maximum depth from GT curve (at dose=255)
        self.depth_max = self.model_gt(torch.tensor(255.0, device=self.device)) * 1e-6

        # Model switches
        self.use_psf = True
        self.use_gt = True

    def forward(
        self,
        field_in: torch.Tensor,
        backward: bool = False,
        cutoff_mtf: float = 0.05
    ) -> torch.Tensor:
        """Apply fabrication model.

        Args:
            field_in: Input dose [B, C, H, W] (forward) or height (backward)
            backward: If True, apply inverse model (height -> dose)
            cutoff_mtf: Cutoff for inverse MTF regularization

        Returns:
            Height [B, C, H, W] (forward) or dose (backward)
        """
        # Recompute kernels if resolution changed
        if self.precomputed_H is None or self.field_resolution != field_in.size():
            self._compute_H(field_in)
            self._compute_H_b(field_in, cutoff_mtf)

        if backward:
            return self._backward(field_in)
        else:
            return self._forward(field_in)

    def _forward(self, dose: torch.Tensor) -> torch.Tensor:
        """Forward model: dose -> height."""
        dose = torch.clamp(dose, 0.0, 255.0)

        if self.use_psf:
            # Apply PSF (MTF low-pass filter)
            dose = torch.fft.fftn(dose, dim=(-2, -1), norm='ortho')
            dose = self.precomputed_H * dose
            dose = torch.fft.ifftn(dose, dim=(-2, -1), norm='ortho').real

        if self.use_gt:
            # Apply GT curve (dose -> depth in um, then convert to m)
            depth = self.model_gt(dose) * 1e-6
        else:
            # Linear approximation
            depth = self.depth_max * (1 - dose / 255.0)

        # Depth to height (height = max_depth - depth)
        height = self.depth_max - depth

        return height

    def _backward(self, height: torch.Tensor) -> torch.Tensor:
        """Backward model: height -> dose."""
        # Height to depth
        depth = self.depth_max - height

        if self.use_gt:
            # Apply inverse GT (depth in um -> dose)
            dose = self.model_gt_inv(depth * 1e6)
        else:
            # Linear approximation
            dose = (1 - depth / self.depth_max) * 255.0

        if self.use_psf:
            # Apply inverse PSF
            dose = torch.fft.fftn(dose, dim=(-2, -1), norm='ortho')
            dose = self.precomputed_H_b * dose
            dose = torch.fft.ifftn(dose, dim=(-2, -1), norm='ortho').real

        return torch.clamp(dose, 0.0, 255.0)

    def gt(self, dose: torch.Tensor) -> torch.Tensor:
        """Apply GT curve only (dose -> height in meters)."""
        depth = self.model_gt(dose) * 1e-6
        return self.depth_max - depth

    def gt_inv(self, height: torch.Tensor) -> torch.Tensor:
        """Apply inverse GT curve (height -> dose)."""
        depth = self.depth_max - height
        return self.model_gt_inv(depth * 1e6)

    def _compute_H(self, field_in: torch.Tensor):
        """Compute forward PSF kernel."""
        self.field_resolution = field_in.size()
        num_y, num_x = self.field_resolution[2], self.field_resolution[3]
        dy, dx = self.feature_size

        # Frequency coordinates
        fy = np.fft.fftfreq(num_y, dy)
        fx = np.fft.fftfreq(num_x, dx)
        FX, FY = np.meshgrid(fx, fy)

        # Radial frequency in um^-1
        F_rad = np.sqrt(FX**2 + FY**2) * 1e-6

        # Apply MTF
        self.precomputed_H = torch.tensor(
            self.model_lp(F_rad), dtype=self.dtype
        )[None, None, :, :].to(self.device)
        self.precomputed_H.requires_grad = False

    def _compute_H_b(self, field_in: torch.Tensor, cutoff_mtf: float = 0.05):
        """Compute backward (inverse) PSF kernel with regularization."""
        self.field_resolution = field_in.size()
        num_y, num_x = self.field_resolution[2], self.field_resolution[3]
        dy, dx = self.feature_size

        fy = np.fft.fftfreq(num_y, dy)
        fx = np.fft.fftfreq(num_x, dx)
        FX, FY = np.meshgrid(fx, fy)
        F_rad = np.sqrt(FX**2 + FY**2) * 1e-6

        M = self.model_lp(F_rad)

        # Regularize small values to avoid amplification
        M[M < cutoff_mtf] = 1.0
        M[num_y // 2, num_x // 2] = 1.0  # DC component

        self.precomputed_H_b = torch.tensor(
            1.0 / M, dtype=self.dtype
        )[None, None, :, :].to(self.device)
        self.precomputed_H_b.requires_grad = False


def create_fabrication_model(
    calibration: 'FabricationCalibration',
    feature_size: tuple,
    device: torch.device = None
) -> FabricationModel:
    """Create fabrication model from calibration data.

    Args:
        calibration: FabricationCalibration config object
        feature_size: (dy, dx) pixel size in meters
        device: Torch device

    Returns:
        FabricationModel instance
    """
    fit = FitModel(
        gt_data=calibration.gt_data,
        lp_data=calibration.lp_data,
        lp_ratio=calibration.lp_ratio,
        device=device
    )
    return FabricationModel(fit, feature_size, device)
