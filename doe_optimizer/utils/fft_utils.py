"""FFT-related utility functions.

Reference: Y. Peng et al., Neural Holography, SIGGRAPH Asia 2020
"""

import numpy as np
import torch


class ZoomFFT2:
    """Compute the chirp z-transform to achieve scaled FFT.

    Input shape: [B, C, H, W]
    Dimension 'C' is used for multi-channel processing.

    Parameters
    ----------
    n : tuple
        Input resolution (ny, nx)
    sy, sx : numpy array
        Frequency zoom factor for each channel
    m : tuple, optional
        Output resolution, defaults to input resolution
    """

    def __init__(
        self,
        n: tuple,
        sy: np.ndarray,
        sx: np.ndarray,
        m: tuple = None,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32
    ):
        if m is None:
            m = n

        # Handle scalar or array zoom factors
        sy = np.atleast_1d(sy)
        sx = np.atleast_1d(sx)
        num_channel = sy.size

        if sy.size != sx.size:
            raise ValueError('Channel number is not equal for x and y')

        sx = sx.ravel()[None, :, None, None]
        sy = sy.ravel()[None, :, None, None]

        num_y, num_x = n[0], n[1]
        num_v, num_u = m[0], m[1]

        # Normalization factor
        self.N = torch.tensor(num_v * num_u).to(device=device, dtype=dtype)

        # Frequency step for CZT: w = -zoom/N (in normalized frequency)
        wx = -2 * np.pi * sx / num_u
        wy = -2 * np.pi * sy / num_v

        # Starting frequency for centered FFT: a = -(N//2)/N * zoom
        # This gives frequencies: a, a+w, a+2w, ..., a+(M-1)w
        # For zoom=1: f_k = -(N//2)/N + k/N = (k - N//2)/N for k=0..N-1
        # Which matches fftshift output: frequencies at k-N//2 for k=0..N-1
        ax = 2 * np.pi * (-(num_x // 2) / num_x) * sx
        ay = 2 * np.pi * (-(num_y // 2) / num_y) * sy

        kky = (np.arange(-num_y + 1, max(num_y, num_v))[None, None, :, None] ** 2) / 2
        kkx = (np.arange(-num_x + 1, max(num_x, num_u))[None, None, None, :] ** 2) / 2
        nny = np.arange(0, num_y)[None, None, :, None]
        nnx = np.arange(0, num_x)[None, None, None, :]

        ww = wx * kkx + wy * kky
        aa = ax * (-nnx) + ay * (-nny) + ww[:, :, (num_y - 1):(2 * num_y - 1), (num_x - 1):(2 * num_x - 1)]

        nfft = (
            int(2 ** np.ceil(np.log2(num_v + num_y - 1))),
            int(2 ** np.ceil(np.log2(num_u + num_x - 1)))
        )

        yidx = slice(num_y - 1, num_y + num_v - 1)
        xidx = slice(num_x - 1, num_x + num_u - 1)

        # Upload chirp signals to device
        # Initial chirp: A
        # FFT kernel: W
        # IFFT kernel: iW
        # Outer chirp: B
        # Phase correction: C
        A = torch.from_numpy(aa).to(device=device, dtype=dtype)
        self._A = torch.exp(1j * A)

        B = torch.from_numpy(ww[:, :, (num_y - 1):(num_v + num_y - 1), (num_x - 1):(num_u + num_x - 1)]).to(
            device=device, dtype=dtype
        )
        self._B = torch.exp(1j * B)

        W = torch.from_numpy(-ww[:, :, :(num_v + num_y - 1), :(num_u + num_x - 1)]).to(device=device, dtype=dtype)
        # Use standard normalization for convolution kernel (not 'ortho')
        # This avoids the sqrt(nfft) scaling issue in Bluestein convolution
        self._W = torch.fft.fft2(torch.exp(1j * W), nfft)
        self._iW = torch.fft.fft2(torch.exp(-1j * W), nfft)

        self._yidx = yidx
        self._xidx = xidx
        self._nfft = nfft

        # Store dimensions for ortho normalization in cfft2/cifft2
        self._num_y = num_y
        self._num_x = num_x
        self._num_v = num_v
        self._num_u = num_u

    def fft2(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the zoomed FFT.

        Computes: sum_{n=0}^{N-1} x_n * exp(-j*2*pi*f*n)

        Uses standard (non-ortho) normalization for correct Bluestein convolution.
        """
        # Standard FFT convolution: ifft(fft(x) * fft(h))
        # Using standard normalization: fft has no scaling, ifft divides by N
        y = torch.fft.ifft2(self._W * torch.fft.fft2(x * self._A, self._nfft))
        y = y[..., self._yidx, self._xidx] * self._B
        return y

    def ifft2(self, x: torch.Tensor) -> torch.Tensor:
        """Calculate the zoomed inverse FFT.

        Computes: sum_{k=0}^{K-1} X_k * exp(j*2*pi*n*fk)
        """
        y = torch.fft.ifft2(self._iW * torch.fft.fft2(x / self._A, self._nfft))
        y = y[..., self._yidx, self._xidx] / self._B
        return y

    def cifft2(self, x: torch.Tensor) -> torch.Tensor:
        """Centered inverse zoom FFT.

        Input should be centered (DC at center).
        Output is also centered (DC at center).
        Uses ortho normalization to match torch.fft conventions.
        """
        # Apply ortho normalization factor
        norm_factor = np.sqrt(self._num_v * self._num_u)
        y = self.ifft2(torch.fft.fftshift(x, dim=(-2, -1))) * norm_factor
        return y

    def cfft2(self, x: torch.Tensor) -> torch.Tensor:
        """Centered zoom FFT.

        Input should be centered (DC at center).
        Output is also centered (DC at center).

        The fft2 method already outputs centered results (DC at center),
        so we only need to apply ifftshift to the input and ortho normalization.
        """
        # Apply ortho normalization: divide by sqrt(N*M)
        norm_factor = np.sqrt(self._num_y * self._num_x)
        y = self.fft2(torch.fft.ifftshift(x, dim=(-2, -1))) / norm_factor
        return y
