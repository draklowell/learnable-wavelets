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

        H_even_new = H_even * cos - H_odd * sin
        H_odd_new = H_even * sin + H_odd * cos
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
        filters = filters.unsqueeze(1).unsqueeze(-1)
        # [1, 2, 1, 1, L]
        filters_T = filters.transpose(-2, -1).transpose(0, 1)

        # [2, 2, 1, L, L]
        out = filters * filters_T

        # [4, 1, L, L]
        return out.view(4, 1, L, L)
