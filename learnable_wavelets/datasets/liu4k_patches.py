from pathlib import Path

import torch
from torch.utils.data import Dataset


def _normalize_split(split: str) -> str:
    normalized = split.strip().lower()
    aliases = {
        "train": "train",
        "training": "train",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "dev": "validation",
    }
    if normalized not in aliases:
        raise ValueError("split must be one of: train, validation")
    return aliases[normalized]


class LIU4KPatchesDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        transform=None,
    ) -> None:
        super().__init__()
        self.split = _normalize_split(split)
        self.root = Path(root) / self.split
        self.transform = transform

        self.samples = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pt"}
        )

        if not self.samples:
            raise RuntimeError(
                f"No prepared LIU4K patch files found in {self.root}. "
                "Run build_liu4k_patches(...) first."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path = self.samples[index]

        sample = torch.load(path)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
