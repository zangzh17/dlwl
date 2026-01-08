"""
Evaluation metrics dataclass.

Provides structured metrics for different DOE types.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
import numpy as np


@dataclass
class EvaluationMetrics:
    """Evaluation metrics for DOE optimization results.

    Contains different metrics relevant to different DOE types:
    - Splitters: order efficiencies, uniformity
    - Diffusers: uniformity, PSNR, SSIM
    - Lenses: encircled energy, Airy disk analysis

    Attributes:
        total_efficiency: Total diffraction efficiency (sum of all orders / total input)
        uniformity: Uniformity metric (1 - (max-min)/(max+min))
        mean_efficiency: Mean efficiency per order/spot
        std_efficiency: Standard deviation of efficiencies
        order_efficiencies: Per-order efficiency list (for splitters)
        psnr: Peak signal-to-noise ratio (dB)
        ssim: Structural similarity index
        encircled_energy: Energy within Airy disk (for lenses)
        spot_positions: Detected spot positions [(y, x), ...]
        doe_type: Type of DOE evaluated
    """
    # Common metrics
    total_efficiency: Optional[float] = None
    uniformity: Optional[float] = None

    # Splitter/spot projector metrics
    mean_efficiency: Optional[float] = None
    std_efficiency: Optional[float] = None
    order_efficiencies: Optional[List[float]] = None
    order_labels: Optional[List[str]] = None

    # Pattern matching metrics (diffuser, custom)
    psnr: Optional[float] = None
    ssim: Optional[float] = None
    mse: Optional[float] = None

    # Lens metrics
    encircled_energy: Optional[float] = None
    airy_radius: Optional[float] = None

    # General info
    spot_positions: Optional[List[tuple]] = None
    doe_type: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        result = {}
        for key, value in self.__dict__.items():
            if value is not None:
                result[key] = value
        return result

    @classmethod
    def for_splitter(
        cls,
        order_efficiencies: List[float],
        order_labels: Optional[List[str]] = None
    ) -> 'EvaluationMetrics':
        """Create metrics for splitter evaluation.

        Args:
            order_efficiencies: Efficiency for each order
            order_labels: Labels like "(-1, 0)", "(0, 0)", etc.

        Returns:
            EvaluationMetrics for splitter
        """
        effs = np.array(order_efficiencies)
        total = float(effs.sum())
        mean = float(effs.mean())
        std = float(effs.std())

        if effs.max() + effs.min() > 0:
            uniformity = 1 - (effs.max() - effs.min()) / (effs.max() + effs.min())
        else:
            uniformity = 0.0

        return cls(
            total_efficiency=total,
            uniformity=float(uniformity),
            mean_efficiency=mean,
            std_efficiency=std,
            order_efficiencies=order_efficiencies,
            order_labels=order_labels,
            doe_type='splitter'
        )

    @classmethod
    def for_diffuser(
        cls,
        target: np.ndarray,
        simulated: np.ndarray,
        roi_mask: Optional[np.ndarray] = None
    ) -> 'EvaluationMetrics':
        """Create metrics for diffuser evaluation.

        Args:
            target: Target intensity pattern
            simulated: Simulated intensity pattern
            roi_mask: Optional ROI mask

        Returns:
            EvaluationMetrics for diffuser
        """
        if roi_mask is not None:
            target = target * roi_mask
            simulated = simulated * roi_mask

        # Normalize
        target = target / (target.sum() + 1e-10)
        simulated = simulated / (simulated.sum() + 1e-10)

        # MSE
        mse = float(((target - simulated) ** 2).mean())

        # PSNR
        max_val = max(target.max(), simulated.max())
        if mse > 0:
            psnr = float(10 * np.log10(max_val ** 2 / mse))
        else:
            psnr = 100.0

        # Uniformity in target region
        if roi_mask is not None:
            sim_roi = simulated[roi_mask > 0]
        else:
            sim_roi = simulated.flatten()

        if sim_roi.max() + sim_roi.min() > 0:
            uniformity = 1 - (sim_roi.max() - sim_roi.min()) / (sim_roi.max() + sim_roi.min())
        else:
            uniformity = 0.0

        # Total efficiency (energy in ROI)
        if roi_mask is not None:
            total_eff = float((simulated * roi_mask).sum())
        else:
            total_eff = 1.0

        return cls(
            total_efficiency=total_eff,
            uniformity=float(uniformity),
            psnr=psnr,
            mse=mse,
            doe_type='diffuser'
        )

    @classmethod
    def for_lens(
        cls,
        intensity: np.ndarray,
        spot_center: tuple,
        airy_radius: float
    ) -> 'EvaluationMetrics':
        """Create metrics for lens evaluation.

        Args:
            intensity: Intensity pattern at focal plane
            spot_center: (y, x) center of focus
            airy_radius: Airy disk radius in pixels

        Returns:
            EvaluationMetrics for lens
        """
        H, W = intensity.shape
        cy, cx = spot_center

        # Create circular mask for Airy disk
        y_coords = np.arange(H) - cy
        x_coords = np.arange(W) - cx
        YY, XX = np.meshgrid(y_coords, x_coords, indexing='ij')
        mask = (YY ** 2 + XX ** 2) <= airy_radius ** 2

        # Encircled energy
        total_energy = intensity.sum()
        airy_energy = intensity[mask].sum()
        encircled = float(airy_energy / (total_energy + 1e-10))

        return cls(
            total_efficiency=encircled,
            encircled_energy=encircled,
            airy_radius=airy_radius,
            spot_positions=[spot_center],
            doe_type='lens'
        )
