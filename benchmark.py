import csv
from itertools import product

import torch
import torchvision
from torchvision.transforms import v2
from tqdm import tqdm

from bench.jpeg import JPEG2000Compressor, JPEGCompressor
from bench.learnable_wavelets import LearnableWaveletsCompressor
from learnable_wavelets.datasets.kodak import KodakDataset


def save(image: torch.Tensor, path: str) -> None:
    image = (image.clamp(-1, 1) + 1) / 2
    image = (image * 255).to(torch.uint8)

    # Use compress_level=0 to avoid any additional compression artifacts from PNG encoding
    torchvision.io.write_png(image.cpu(), path, compression_level=0)


def psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    mse = torch.mean((original - reconstructed) ** 2).item()

    if mse == 0:
        return float("inf")

    max_pixel = 2.0
    return 20 * torch.log10(max_pixel / torch.sqrt(torch.tensor(mse))).item()


def main():
    dataset = KodakDataset(
        "datasets/kodak",
        transform=v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Grayscale(),
                v2.Lambda(lambda x: x * 2 - 1),
            ]
        ),
    )

    compressors = (
        [
            (
                "lw",
                "lw",
                LearnableWaveletsCompressor(
                    "./params/config.yaml", "./params/state.pt"
                ),
            ),
        ]
        + [
            ("jpeg", q, JPEGCompressor(quality=q))
            for q in [10, 20, 30, 40, 50, 60, 70, 80, 90]
        ]
        + [
            ("jpeg2000", r, JPEG2000Compressor(rate=r))
            for r in [0.1, 0.2, 0.3, 0.4, 0.5]
        ]
    )

    results = []
    for (compressor_name, param, compressor), (image, image_name) in tqdm(
        product(compressors, dataset), total=len(compressors) * len(dataset)
    ):
        bytes_, reconstructed = compressor.compress(image)

        save(reconstructed, f"results/{compressor_name}_{image_name}_{param}.png")

        results.append(
            {
                "param": param,
                "compressor": compressor_name,
                "name": image_name,
                "pixels": image.numel(),
                "bytes": bytes_,
                "psnr": psnr(image, reconstructed),
            }
        )

    with open("results/report.csv", "w") as f:
        writer = csv.DictWriter(
            f, fieldnames=["compressor", "name", "pixels", "bytes", "psnr", "param"]
        )
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
