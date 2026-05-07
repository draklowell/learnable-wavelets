from abc import ABC, abstractmethod

import torch


class BaseCompressor(ABC):
    @abstractmethod
    def compress(self, image: torch.Tensor) -> tuple[int, torch.Tensor]:
        # All images are of the shape (1, H, W) and in the range [-1.0, 1.0]
        pass
