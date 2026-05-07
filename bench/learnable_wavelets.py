import constriction
import numpy as np
import torch

from bench.base import BaseCompressor
from learnable_wavelets.config import load_config
from learnable_wavelets.module import WaveletModule


def estimate_laplace_pmf(x: np.ndarray) -> np.ndarray | None:
    if x.size == 0:
        return None

    mu = np.mean(x).astype(np.float64).item()
    b = np.mean(np.abs(x - mu)).astype(np.float64)
    b = max(b, 1e-3)

    symbols = np.arange(0, 256).astype(np.float64)

    pmf = np.exp(-np.abs(symbols - mu) / b)
    pmf = pmf / pmf.sum()
    pmf = pmf.astype(np.float64)

    return pmf


def entropy_code(x: np.ndarray) -> int:
    x = x.ravel()

    pmf = estimate_laplace_pmf(x)  # estimated
    actual = np.bincount(x, minlength=256) / x.size

    if pmf is None:
        # 2 bytes to encode mu and b
        return 2

    x = x.astype(np.int32)

    model = constriction.stream.model.Categorical(pmf, perfect=False)

    encoder = constriction.stream.queue.RangeEncoder()
    encoder.encode(x, model)

    compressed = encoder.get_compressed()

    return compressed.nbytes + 2  # +2 bytes to encode mu and b


class LearnableWaveletsCompressor(BaseCompressor):
    def __init__(self, config_path: str, state_path: str):
        self.config_path = config_path
        self.state_path = state_path

        self.config = load_config(config_path)
        self.module = WaveletModule(self.config)

        self.module.load_state_dict(torch.load(state_path, map_location="cpu"))

    def compress(self, image: torch.Tensor) -> tuple[int, torch.Tensor]:
        self.module.eval()

        bytes_ = 0

        def intercept(x: torch.Tensor) -> torch.Tensor:
            nonlocal bytes_
            bytes_ += entropy_code(x.byte().cpu().numpy())

            return x

        image = image.unsqueeze(0)
        with torch.inference_mode():
            output = self.module(image, middleware=intercept)
        output = output.squeeze(0)

        return bytes_, output
