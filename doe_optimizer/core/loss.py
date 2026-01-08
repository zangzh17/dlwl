"""
Pluggable loss function interface.

This module provides:
- BaseLoss: Abstract base class for loss functions
- Standard losses: L1Loss, L2Loss
- Specialized losses: FocalEfficiencyLoss, FocalUniformityLoss
- CompositeLoss: Weighted combination of multiple losses
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Tuple
import torch
import numpy as np


class BaseLoss(ABC):
    """Base class for loss functions.

    All loss functions take:
    - pred: Predicted amplitude/intensity [B, C, H, W]
    - target: Target amplitude [B, C, H, W]
    - mask: Optional ROI mask [H, W] or [B, C, H, W]

    Returns:
    - Scalar loss value (lower is better)
    """

    @abstractmethod
    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute loss value.

        Args:
            pred: Predicted amplitude [B, C, H, W]
            target: Target amplitude [B, C, H, W]
            mask: Optional ROI mask

        Returns:
            Scalar loss tensor
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Loss function name."""
        pass

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Call compute method."""
        return self.compute(pred, target, mask)


class L1Loss(BaseLoss):
    """Mean Absolute Error loss."""

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        diff = torch.abs(pred - target)
        if mask is not None:
            diff = diff * mask
            return diff.sum() / (mask.sum() + 1e-10)
        return diff.mean()

    @property
    def name(self) -> str:
        return "L1"


class L2Loss(BaseLoss):
    """Mean Squared Error loss."""

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        diff = (pred - target) ** 2
        if mask is not None:
            diff = diff * mask
            return diff.sum() / (mask.sum() + 1e-10)
        return diff.mean()

    @property
    def name(self) -> str:
        return "L2"


class FocalEfficiencyLoss(BaseLoss):
    """Maximize energy within Airy disk of each focal spot.

    Used for lens and spot array optimization where we want to
    maximize the encircled energy at specific locations.
    """

    def __init__(
        self,
        spot_positions: List[Tuple[int, int]],
        airy_radius: float
    ):
        """Initialize focal efficiency loss.

        Args:
            spot_positions: List of (y, x) positions for each spot
            airy_radius: Airy disk radius in pixels
        """
        self.spot_positions = spot_positions
        self.airy_radius = airy_radius
        self._masks = None

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute negative focal efficiency (for minimization)."""
        B, C, H, W = pred.shape

        # Create spot masks if not cached
        if self._masks is None or self._masks.shape[-2:] != (H, W):
            self._masks = self._create_spot_masks(H, W, pred.device, pred.dtype)

        # Compute intensity
        intensity = pred.abs() ** 2

        # Sum intensity within each spot's Airy disk
        total_efficiency = 0.0
        for spot_mask in self._masks:
            spot_intensity = (intensity * spot_mask).sum()
            total_efficiency = total_efficiency + spot_intensity

        # Normalize by total intensity
        total_intensity = intensity.sum() + 1e-10
        efficiency = total_efficiency / total_intensity

        # Return negative (we want to maximize efficiency)
        return -efficiency

    def _create_spot_masks(
        self,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> List[torch.Tensor]:
        """Create circular masks for each spot."""
        masks = []
        y_coords = torch.arange(H, device=device, dtype=dtype)
        x_coords = torch.arange(W, device=device, dtype=dtype)
        YY, XX = torch.meshgrid(y_coords, x_coords, indexing='ij')

        for py, px in self.spot_positions:
            dist_sq = (YY - py) ** 2 + (XX - px) ** 2
            mask = (dist_sq <= self.airy_radius ** 2).float()
            masks.append(mask.unsqueeze(0).unsqueeze(0))  # [1, 1, H, W]

        return masks

    @property
    def name(self) -> str:
        return "FocalEfficiency"


class FocalUniformityLoss(BaseLoss):
    """Minimize variance of focal spot efficiencies.

    Used when we want uniform intensity across all spots.
    """

    def __init__(
        self,
        spot_positions: List[Tuple[int, int]],
        airy_radius: float
    ):
        """Initialize focal uniformity loss.

        Args:
            spot_positions: List of (y, x) positions
            airy_radius: Airy disk radius in pixels
        """
        self.spot_positions = spot_positions
        self.airy_radius = airy_radius
        self._masks = None

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute efficiency variance."""
        B, C, H, W = pred.shape

        # Create spot masks if not cached
        if self._masks is None or self._masks.shape[-2:] != (H, W):
            self._masks = self._create_spot_masks(H, W, pred.device, pred.dtype)

        # Compute intensity
        intensity = pred.abs() ** 2
        total_intensity = intensity.sum() + 1e-10

        # Compute efficiency for each spot
        efficiencies = []
        for spot_mask in self._masks:
            spot_intensity = (intensity * spot_mask).sum()
            efficiency = spot_intensity / total_intensity
            efficiencies.append(efficiency)

        # Stack and compute variance
        effs = torch.stack(efficiencies)
        mean_eff = effs.mean()
        variance = ((effs - mean_eff) ** 2).mean()

        return variance

    def _create_spot_masks(
        self,
        H: int,
        W: int,
        device: torch.device,
        dtype: torch.dtype
    ) -> List[torch.Tensor]:
        """Create circular masks for each spot."""
        masks = []
        y_coords = torch.arange(H, device=device, dtype=dtype)
        x_coords = torch.arange(W, device=device, dtype=dtype)
        YY, XX = torch.meshgrid(y_coords, x_coords, indexing='ij')

        for py, px in self.spot_positions:
            dist_sq = (YY - py) ** 2 + (XX - px) ** 2
            mask = (dist_sq <= self.airy_radius ** 2).float()
            masks.append(mask.unsqueeze(0).unsqueeze(0))

        return masks

    @property
    def name(self) -> str:
        return "FocalUniformity"


class CompositeLoss(BaseLoss):
    """Weighted combination of multiple loss functions."""

    def __init__(self, losses: Dict[BaseLoss, float]):
        """Initialize composite loss.

        Args:
            losses: Dictionary mapping loss functions to weights
        """
        self.losses = losses

    def compute(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute weighted sum of losses."""
        total = 0.0
        for loss_fn, weight in self.losses.items():
            total = total + weight * loss_fn.compute(pred, target, mask)
        return total

    @property
    def name(self) -> str:
        parts = [f"{w:.2f}*{l.name}" for l, w in self.losses.items()]
        return " + ".join(parts)


def create_loss(loss_type: str, **kwargs) -> BaseLoss:
    """Factory function for loss creation.

    Args:
        loss_type: Loss type string ('L1', 'L2', 'focal_efficiency', etc.)
        **kwargs: Additional arguments for specific loss types

    Returns:
        Loss function instance

    Raises:
        ValueError: If loss_type is not recognized
    """
    loss_type = loss_type.lower()

    if loss_type == 'l1':
        return L1Loss()

    elif loss_type == 'l2':
        return L2Loss()

    elif loss_type == 'focal_efficiency':
        spot_positions = kwargs.get('spot_positions')
        airy_radius = kwargs.get('airy_radius', 3.0)
        if spot_positions is None:
            raise ValueError("FocalEfficiencyLoss requires spot_positions")
        return FocalEfficiencyLoss(spot_positions, airy_radius)

    elif loss_type == 'focal_uniformity':
        spot_positions = kwargs.get('spot_positions')
        airy_radius = kwargs.get('airy_radius', 3.0)
        if spot_positions is None:
            raise ValueError("FocalUniformityLoss requires spot_positions")
        return FocalUniformityLoss(spot_positions, airy_radius)

    elif loss_type == 'composite':
        loss_configs = kwargs.get('losses')
        if loss_configs is None:
            raise ValueError("CompositeLoss requires losses dict")
        losses = {}
        for config, weight in loss_configs.items():
            if isinstance(config, str):
                losses[create_loss(config, **kwargs)] = weight
            else:
                losses[config] = weight
        return CompositeLoss(losses)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}. "
                         f"Supported: L1, L2, focal_efficiency, focal_uniformity, composite")
