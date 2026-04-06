import io
import re
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _is_image_file(name: str) -> bool:
    return Path(name).suffix.lower() in IMAGE_EXTENSIONS


def _split_part_index(path: Path) -> int:
    m = re.fullmatch(r"\.z(\d+)", path.suffix.lower())
    if not m:
        raise ValueError(f"Not a split part: {path}")
    return int(m.group(1))


def _find_split_parts(zip_path: Path) -> list[Path]:
    stem = zip_path.with_suffix("")
    parts = []
    for candidate in zip_path.parent.glob(f"{stem.name}.z*"):
        if re.fullmatch(r"\.z\d+", candidate.suffix.lower()):
            parts.append(candidate)
    return sorted(parts, key=_split_part_index)


def _build_combined_zip(
    split_parts: list[Path], last_zip_part: Path, out_zip: Path
) -> Path:
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    latest_src_mtime = max(
        [p.stat().st_mtime for p in split_parts] + [last_zip_part.stat().st_mtime]
    )
    if out_zip.exists() and out_zip.stat().st_mtime >= latest_src_mtime:
        return out_zip

    with out_zip.open("wb") as out_f:
        for part in split_parts:
            with part.open("rb") as in_f:
                while True:
                    chunk = in_f.read(1024 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)

        with last_zip_part.open("rb") as in_f:
            while True:
                chunk = in_f.read(1024 * 1024)
                if not chunk:
                    break
                out_f.write(chunk)

    return out_zip


def _to_grayscale_tensor(image: Image.Image) -> torch.Tensor:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = rgb.mean(axis=2)
    return torch.from_numpy(gray)


class LIU4KDataset(Dataset):

    def __init__(
        self,
        root: str | Path,
        recursive: bool = True,
        return_class: bool = False,
        cache_dir: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root}")

        self.recursive = recursive
        self.return_class = return_class
        self.cache_dir = (
            Path(cache_dir) if cache_dir is not None else (self.root / ".liu4k_cache")
        )

        self.samples: list[tuple[str, str, int]] = []

        self.class_to_idx: dict[str, int] = {}
        self.classes: list[str] = []

        self._zip_handles: dict[Path, ZipFile] = {}

        self._index_dataset()

        if not self.samples:
            raise RuntimeError(f"No images found in {self.root}")

    def _class_index(self, class_name: str) -> int:
        if class_name not in self.class_to_idx:
            self.class_to_idx[class_name] = len(self.classes)
            self.classes.append(class_name)
        return self.class_to_idx[class_name]

    def _resolve_archive(self, zip_path: Path) -> Path:
        split_parts = _find_split_parts(zip_path)
        if not split_parts:
            return zip_path

        combined_name = f"{zip_path.stem}__combined.zip"
        combined_path = self.cache_dir / combined_name
        return _build_combined_zip(split_parts, zip_path, combined_path)

    def _iter_paths(self) -> list[Path]:
        if self.recursive:
            files = [p for p in self.root.rglob("*") if p.is_file()]
        else:
            files = [p for p in self.root.iterdir() if p.is_file()]

        cache_dir_resolved = self.cache_dir.resolve()
        out = []
        for p in files:
            try:
                p.resolve().relative_to(cache_dir_resolved)
                continue
            except ValueError:
                out.append(p)
        return out

    def _index_dataset(self) -> None:
        all_files = self._iter_paths()

        for p in all_files:
            if _is_image_file(p.name):
                class_name = p.parent.name
                class_idx = self._class_index(class_name)
                self.samples.append(("file", str(p.resolve()), class_idx))

        zip_files = [p for p in all_files if p.suffix.lower() == ".zip"]
        for zip_path in zip_files:
            class_name = zip_path.stem
            class_idx = self._class_index(class_name)

            real_zip = self._resolve_archive(zip_path)
            with ZipFile(real_zip, "r") as zf:
                for name in zf.namelist():
                    if not name.endswith("/") and _is_image_file(name):
                        ref = f"{real_zip.resolve()}::{name}"
                        self.samples.append(("zip", ref, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def _get_zip(self, path: Path) -> ZipFile:
        if path not in self._zip_handles:
            self._zip_handles[path] = ZipFile(path, "r")
        return self._zip_handles[path]

    def __getitem__(self, index: int) -> torch.Tensor | tuple[torch.Tensor, int]:
        source_type, source_ref, class_idx = self.samples[index]

        if source_type == "file":
            with Image.open(source_ref) as img:
                x = _to_grayscale_tensor(img)
        else:
            zip_path_str, member_name = source_ref.split("::", 1)
            zip_path = Path(zip_path_str)
            zf = self._get_zip(zip_path)
            with zf.open(member_name, "r") as f:
                raw = f.read()
            with Image.open(io.BytesIO(raw)) as img:
                x = _to_grayscale_tensor(img)

        if self.return_class:
            return x, class_idx
        return x

    def close(self) -> None:
        for zf in self._zip_handles.values():
            zf.close()
        self._zip_handles.clear()

    def __del__(self) -> None:
        self.close()


def make_liu4k_dataloader(
    root: str | Path,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    recursive: bool = True,
    return_class: bool = False,
    cache_dir: str | Path | None = None,
) -> DataLoader:
    dataset = LIU4KDataset(
        root=root,
        recursive=recursive,
        return_class=return_class,
        cache_dir=cache_dir,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


# example usage:
# train_loader = make_liu4k_dataloader(
#    root="path/to/my_dataset",  # path to dataset root directory
#    batch_size=32,              # number of images in each batch
#    shuffle=True,               # shuffle data
#    num_workers=4,              # number of parallel processes for loading
#    pin_memory=True,            # speeds up data transfer to GPU
#    return_class=True           # return (images, class_indices). If False - returns only images
# )

# returns tensor of dimensions (batch_size, height, width)
# with pixel values in [0, 1]
