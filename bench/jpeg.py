import io

import numpy as np
import torch
from PIL import Image

from bench.base import BaseCompressor


def to_pil(image: torch.Tensor) -> Image.Image:
    image = image.clamp(-1, 1)
    image = (image + 1) / 2
    image = (image * 255).byte().cpu().numpy()

    return Image.fromarray(image.squeeze())


def to_tensor(image: Image.Image) -> torch.Tensor:
    image = np.array(image)
    image = torch.from_numpy(image).float()
    image = image / 255.0 * 2 - 1
    return image.unsqueeze(0)


class JPEGCompressor(BaseCompressor):
    def __init__(self, quality: int):
        self.quality = quality

    def compress(self, image: torch.Tensor) -> tuple[int, torch.Tensor]:
        pil = to_pil(image)

        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=self.quality)
        size_bytes = buf.tell()

        buf.seek(0)
        recon = Image.open(buf).convert("L")

        return size_bytes, to_tensor(recon)


class JPEG2000Compressor(BaseCompressor):
    def __init__(self, rate: float):
        self.rate = rate

    def compress(self, image: torch.Tensor) -> tuple[int, torch.Tensor]:
        pil = to_pil(image)

        buf = io.BytesIO()
        pil.save(
            buf, format="JPEG2000", quality_mode="rates", quality_layers=[self.rate]
        )
        size_bytes = buf.tell()

        buf.seek(0)
        recon = Image.open(buf).convert("L")

        return size_bytes, to_tensor(recon)
