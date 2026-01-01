"""Custom pattern generator from user-provided images."""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .base import PatternGenerator
from ..core.config import DOEConfig
from ..utils.image_utils import pad_image


class CustomPatternGenerator(PatternGenerator):
    """Generate target patterns from custom images.

    Supports loading from numpy array, PIL Image, or file path.
    """

    def generate(self) -> torch.Tensor:
        """Generate target pattern from custom image.

        Returns:
            Target amplitude tensor [1, C, H, W]
        """
        target = self.config.target

        # Get target image
        if target.target_image is None:
            raise ValueError("target_image must be provided for custom pattern")

        image = target.target_image

        # Handle different input types
        if isinstance(image, str):
            # Load from file path
            pattern = self._load_from_file(image)
        elif isinstance(image, Image.Image):
            # Convert PIL Image
            pattern = self._from_pil(image)
        elif isinstance(image, np.ndarray):
            # Convert numpy array
            pattern = self._from_numpy(image)
        elif isinstance(image, torch.Tensor):
            pattern = image.to(device=self.device, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Ensure correct dimensions [1, C, H, W]
        if pattern.ndim == 2:
            pattern = pattern.unsqueeze(0).unsqueeze(0)
        elif pattern.ndim == 3:
            pattern = pattern.unsqueeze(0)

        # Resize to target resolution
        target_res = target.target_resolution or self.output_resolution
        pattern = F.interpolate(pattern, size=target_res, mode='bilinear', align_corners=False)

        # Pad to ROI resolution if needed
        roi_res = self.output_resolution
        if pattern.shape[-2:] != roi_res:
            pattern = pad_image(pattern, roi_res)

        # Expand to multi-channel if needed
        if pattern.shape[1] == 1 and self.num_channels > 1:
            pattern = pattern.repeat(1, self.num_channels, 1, 1)

        # Linearize gamma (assume sRGB input)
        pattern = self._srgb_to_linear(pattern)

        # Convert intensity to amplitude
        pattern = torch.sqrt(torch.clamp(pattern, 0, 1))

        return self.normalize(pattern)

    def _load_from_file(self, path: str) -> torch.Tensor:
        """Load image from file path.

        Args:
            path: File path to image

        Returns:
            Image tensor [1, C, H, W]
        """
        img = Image.open(path)
        return self._from_pil(img)

    def _from_pil(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL Image to tensor.

        Args:
            image: PIL Image

        Returns:
            Image tensor [1, C, H, W]
        """
        # Convert to grayscale if color
        if image.mode != 'L':
            image = image.convert('L')

        # Convert to tensor
        arr = np.array(image, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).to(device=self.device)

    def _from_numpy(self, arr: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor.

        Args:
            arr: Numpy array (grayscale or RGB)

        Returns:
            Image tensor [1, C, H, W]
        """
        # Normalize to [0, 1] if needed
        if arr.max() > 1.0:
            arr = arr.astype(np.float32) / 255.0
        else:
            arr = arr.astype(np.float32)

        # Convert RGB to grayscale
        if arr.ndim == 3 and arr.shape[-1] == 3:
            arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

        return torch.from_numpy(arr).to(device=self.device)

    def _srgb_to_linear(self, img: torch.Tensor) -> torch.Tensor:
        """Convert sRGB to linear color space.

        Args:
            img: sRGB image tensor

        Returns:
            Linear image tensor
        """
        # sRGB to linear conversion
        threshold = 0.04045
        linear = torch.where(
            img <= threshold,
            img / 12.92,
            ((img + 0.055) / 1.055) ** 2.4
        )
        return linear
