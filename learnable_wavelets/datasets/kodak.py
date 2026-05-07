from pathlib import Path

import torchvision
from torch.utils.data import Dataset


def _normalize_split(split: str) -> str:
    normalized = split.strip().lower()
    aliases = {
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "dev": "validation",
    }
    if normalized not in aliases:
        raise ValueError("split must be one of: validation")
    return aliases[normalized]


class KodakDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "validation",
        transform=None,
    ) -> None:
        super().__init__()
        self.split = _normalize_split(split)
        self.root = Path(root)
        self.transform = transform

        self.samples = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in {".png"}
        )

        if not self.samples:
            raise RuntimeError(f"No prepared Kodak files found in {self.root}. ")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path = self.samples[index]

        sample = torchvision.io.decode_image(str(path))

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, path.name.removesuffix(".png")
