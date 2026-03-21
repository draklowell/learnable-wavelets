import torch
import torch.nn as nn


# Parametrizing all possible FIR filters for fixed size using polyphase rotations
# in N/2 dimensions to keep the wavelet orthonormal.
# QMF condition for orthonormal wavelets: g[n] = (-1)^n h[N-1-n].
@torch.compile
def compute_filters(angles, dtype=None, device=None):
    dtype = dtype or angles.dtype
    device = device or angles.device

    H_even = torch.zeros(angles.shape[0], dtype=dtype, device=device)
    H_even[0] = 1
    H_odd = torch.zeros(angles.shape[0], dtype=dtype, device=device)

    for theta in angles:
        H_odd = torch.roll(H_odd, 1)

        cos = torch.cos(theta)
        sin = torch.sin(theta)

        H_even_new = H_even * cos + H_odd * sin
        H_odd_new = -H_even * sin + H_odd * cos
        H_even, H_odd = H_even_new, H_odd_new

    low = torch.empty(angles.shape[0] * 2, dtype=dtype, device=device)
    low[::2] = H_even
    low[1::2] = H_odd

    high = low.flip(0).conj()
    high[0::2].neg_()

    # Flip, because torch computes cross-correlation instead of convolution
    high = high.flip(0)
    low = low.flip(0)

    return torch.stack([low, high], dim=0).unsqueeze(1)


class WaveletTransformParameters(nn.Module):
    def __init__(self, support_size: int):
        assert support_size % 2 == 0, "Support size must be even."
        assert support_size > 0, "Support size must be positive."

        super().__init__()
        self.support_size = support_size
        self.angles = nn.Parameter(torch.randn(support_size // 2))

    def forward(self):
        """
        Returns:
            shape (2, 1, L), synthesis filters, where the first dimension corresponds to low/high pass.
        """
        return compute_filters(
            self.angles,
            dtype=self.angles.dtype,
            device=self.angles.device,
        )


class WaveletTransformAnalysis1D(nn.Module):
    def __init__(self, padding_mode: str = "reflect"):
        super().__init__()
        self.padding_mode = padding_mode

    def forward(self, x, filters):
        """
        Args:
            x: shape (B, 1, N).
            filters: shape (2, 1, L), synthesis filters from WaveletTransformParameters.

        Returns:
            tuple
              - shape (B, (N + L - 1) // 2), low-pass coefficients
              - shape (B, (N + L - 1) // 2), high-pass coefficients
        """
        L = filters.shape[-1]
        x_pad = torch.nn.functional.pad(x, (L - 1, L - 1), mode=self.padding_mode)

        y = torch.nn.functional.conv1d(
            x_pad,
            filters,
            stride=2,
            padding=0,
        )

        approx = y[:, 0:1, :]
        details = y[:, 1:2, :]
        return approx, details


class WaveletTransformSynthesis1D(nn.Module):
    def forward(self, low, high, filters):
        """
        Args:
            low: shape (B, 1, (N + L - 1) // 2), low-pass coefficients.
            high: shape (B, 1, (N + L - 1) // 2), high-pass coefficients.
            filters: shape (2, 1, L), synthesis filters from WaveletTransformParameters.

        Returns:
            shape (B, 1, (N + 1)//2*2), reconstructed signal.
        """
        L = filters.shape[-1]
        x_pad = torch.nn.functional.conv_transpose1d(
            torch.cat([low, high], dim=1),
            filters,
            stride=2,
            output_padding=1,
        )

        return x_pad[:, :, L - 1 : -L + 1]


class WaveletTransformAnalysisMultiLevel1D(nn.Module):
    def __init__(self, max_level: None | int = None, padding_mode: str = "reflect"):
        super().__init__()
        self.max_level = max_level
        self.wavelet_transform = WaveletTransformAnalysis1D(padding_mode=padding_mode)

    def forward(self, x, filters):
        L = filters.shape[-1]

        decomposition = []
        level = 0
        approx = x
        while True:
            if self.max_level and level == self.max_level:
                break

            approx, details = self.wavelet_transform(approx, filters)
            decomposition.append(details)

            if approx.shape[-1] < L:
                break

            level += 1

        decomposition.append(approx)
        return decomposition[::-1]


class WaveletTransformSynthesisMultiLevel1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.wavelet_synthesis = WaveletTransformSynthesis1D()

    def forward(self, decomposition, filters):
        approx = decomposition[0]
        for details in decomposition[1:]:
            approx = approx[:, :, : details.shape[-1]]
            approx = self.wavelet_synthesis(approx, details, filters)

        return approx

class WaveletTransformParameters2D(nn.Module):
    def forward(self, filters):
        """
        Args:
            filters: shape (2, 1, L), synthesis filters from WaveletTransformParameters
        Returns:
            shape (4, 1, L, L), synthesis filters for 2D wavelet transform
        """
        L = filters.shape[-1]
        # [2, 1, 1, L, 1]
        filters  = filters.unsqueeze(1).unsqueeze(-1)
        # [1, 2, 1, 1, L]
        filters_T = filters.transpose(-2, -1).transpose(0, 1)

        # [2, 2, 1, L, L]
        out = filters * filters_T

        # [4, 1, L, L]
        return out.view(4, 1, L, L)

class WaveletTransformAnalysis2D(nn.Module):
    def __init__(self, padding_mode: str = "reflect"):
        super().__init__()
        self.padding_mode = padding_mode

    def forward(self, x, filters):
        """
        Args:
            x: shape (B, 1, N, M).
            filters: shape (4, 1, L, L), synthesis filters from WaveletTransformParameters2D.

        Returns:
            tuple
              - shape (B, (N + L - 1) // 2, (M + L - 1) // 2), low-low coefficients
              - shape (B, 3, (N + L - 1) // 2, (M + L - 1) // 2), (low-high, high-low, high-high) coefficients
        """
        L = filters.shape[-1]
        x_pad = torch.nn.functional.pad(x, (L - 1, L - 1, L - 1, L - 1), mode=self.padding_mode)

        y = torch.nn.functional.conv2d(
            x_pad,
            filters,
            stride=(2, 2),
            padding=(0, 0),
        )

        approx = y[:, 0:1, :]
        details = y[:, 1:, :]
        return approx, details


class WaveletTransformSynthesis2D(nn.Module):
    def forward(self, approx, details, filters):
        """
        Args:
            approx: shape (B, 1, (N + L - 1) // 2, (M + L - 1) // 2), approximation coefficients.
            details: shape (B, 3, (N + L - 1) // 2, (M + L - 1) // 2), detail coefficients.
            filters: shape (4, 1, L, L), synthesis filters from WaveletTransformParameters2D.

        Returns:
            shape (B, 1, (N + L - 1) // 2 * 2, (M + L - 1) // 2 * 2), reconstructed signal.
        """
        L = filters.shape[-1]
        x_pad = torch.nn.functional.conv_transpose2d(
            torch.cat([approx, details], dim=1),
            filters,
            stride=(2, 2),
            output_padding=(1, 1),
        )

        return x_pad[:, :, L - 1 : -L + 1, L - 1 : -L + 1]


class WaveletTransformAnalysisMultiLevel2D(nn.Module):
    def __init__(self, max_level: None | int = None, padding_mode: str = "reflect"):
        super().__init__()
        self.max_level = max_level
        self.wavelet_transform = WaveletTransformAnalysis2D(padding_mode=padding_mode)

    def forward(self, x, filters):
        L = filters.shape[-1]

        decomposition = []
        level = 0
        approx = x
        while True:
            if self.max_level and level == self.max_level:
                break

            approx, details = self.wavelet_transform(approx, filters)
            decomposition.append(details)

            if approx.shape[-1] < L or approx.shape[-2] < L:
                break

            level += 1

        decomposition.append(approx)
        return decomposition[::-1]


class WaveletTransformSynthesisMultiLevel2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.wavelet_synthesis = WaveletTransformSynthesis2D()

    def forward(self, decomposition, filters):
        approx = decomposition[0]
        for details in decomposition[1:]:
            approx = approx[:, :, : details.shape[-2], : details.shape[-1]]
            approx = self.wavelet_synthesis(approx, details, filters)

        return approx