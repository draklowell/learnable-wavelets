import torch
from torch import nn

from learnable_wavelets.config import LeafNode, ModuleConfig, SplitNode
from learnable_wavelets.model import (
    WaveletTransformAnalysis2D,
    WaveletTransformParameters,
    WaveletTransformParameters2D,
    WaveletTransformSynthesis2D,
)


def build_module(config: LeafNode | SplitNode):
    if isinstance(config, LeafNode):
        return LeafModule(config)
    elif isinstance(config, SplitNode):
        return SplitModule(config)
    else:
        raise ValueError(f"Invalid config type: {type(config)}")


class LeafModule(nn.Module):
    def __init__(self, type_: LeafNode) -> None:
        super().__init__()
        self.type = type_

    def forward(self, x, _: dict[str, WaveletTransformParameters2D]):
        if self.type == LeafNode.KEEP:
            if self.training:
                return x
            else:
                # Store coefficients in 8 bits
                min_ = torch.min(x)
                max_ = torch.max(x)
                if min_ == max_:
                    return x
                y = torch.round((x - min_) / (max_ - min_) * 255)
                return y / 255 * (max_ - min_) + min_
        elif self.type == LeafNode.DROP:
            return x * 0
        else:
            raise ValueError(f"Invalid leaf node type: {self.type}")


class SplitModule(nn.Module):
    def __init__(self, config: SplitNode) -> None:
        super().__init__()
        self.wavelet = config.wavelet
        self.padding = config.padding
        self.analysis = WaveletTransformAnalysis2D(self.padding)
        self.synthesis = WaveletTransformSynthesis2D()
        self.hh = build_module(config.hh)
        self.hl = build_module(config.hl)
        self.lh = build_module(config.lh)
        self.ll = build_module(config.ll)

    def forward(self, x, filters: dict[str, WaveletTransformParameters2D]):
        ll, details = self.analysis(x, filters[self.wavelet])
        lh = details[:, 0:1, :, :]
        hl = details[:, 1:2, :, :]
        hh = details[:, 2:3, :, :]

        ll = self.ll.forward(ll, filters)
        lh = self.lh.forward(lh, filters)
        hl = self.hl.forward(hl, filters)
        hh = self.hh.forward(hh, filters)

        min_width = min(ll.shape[-1], lh.shape[-1], hl.shape[-1], hh.shape[-1])
        min_height = min(ll.shape[-2], lh.shape[-2], hl.shape[-2], hh.shape[-2])

        ll = ll[:, :, :min_height, :min_width]
        lh = lh[:, :, :min_height, :min_width]
        hl = hl[:, :, :min_height, :min_width]
        hh = hh[:, :, :min_height, :min_width]

        details = torch.cat([hl, hh, lh], dim=1)
        return self.synthesis(ll, details, filters[self.wavelet])


class WaveletModule(nn.Module):
    def __init__(self, config: ModuleConfig) -> None:
        super().__init__()
        self.model = build_module(config.tree)
        self.wavelets = nn.ModuleDict(
            {
                wavelet.name: WaveletTransformParameters(wavelet.support_size)
                for wavelet in config.wavelets
            }
        )
        self.params2d = WaveletTransformParameters2D()

    def forward(self, x):
        filters = {
            name: self.params2d(params()).to(dtype=x.dtype)
            for name, params in self.wavelets.items()
        }
        x_rec = self.model.forward(x, filters)
        return x_rec[:, :, : x.shape[-2], : x.shape[-1]]
