import csv
import importlib.util
import pickle
import random
import struct
import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _load_liu4k_dataset_class():
    liu4k_path = Path(__file__).with_name("liu4k.py")
    spec = importlib.util.spec_from_file_location("liu4k_dataset_module", liu4k_path)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError(f"Failed to load LIU4K module from: {liu4k_path}")
    spec.loader.exec_module(module)
    return module.LIU4KDataset


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _collect_image_files(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file() and _is_image_file(p)]


def _directory_has_any_files(root: Path) -> bool:
    if not root.exists():
        return False
    try:
        next(root.iterdir())
        return True
    except StopIteration:
        return False


def _is_supported_archive(path: Path) -> bool:
    name = path.name.lower()
    return (
        path.suffix.lower() == ".zip"
        or name.endswith(".tar")
        or name.endswith(".tar.gz")
        or name.endswith(".tgz")
    )


def _directory_has_non_archive_files(root: Path) -> bool:
    if not root.exists():
        return False
    for path in root.rglob("*"):
        if path.is_file() and not _is_supported_archive(path):
            return True
    return False


def _extract_supported_archives(root: Path) -> None:
    if not root.exists():
        return

    archives = [
        p
        for p in root.iterdir()
        if p.is_file()
        and (
            _is_supported_archive(p)
        )
    ]

    for archive_path in archives:
        if archive_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(root)
            continue

        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(root)


def _to_gray_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = arr.mean(axis=2)
    return torch.from_numpy(gray)


def _load_image_tensor(path: Path) -> torch.Tensor:
    with Image.open(path) as img:
        return _to_gray_tensor(img)


def _try_kaggle_download(dataset_slug: str) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "Kaggle fallback requires `kagglehub`. Install it with: pip install kagglehub"
        ) from exc

    downloaded_path = kagglehub.dataset_download(dataset_slug)
    return Path(downloaded_path)


def _prepare_dataset_root(
    root: Path,
    *,
    auto_extract_archives: bool,
    kaggle_dataset_slug: str | None,
    enable_kaggle_fallback: bool,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)

    if _directory_has_non_archive_files(root):
        return root

    if auto_extract_archives:
        _extract_supported_archives(root)
        if _directory_has_non_archive_files(root):
            return root

    if enable_kaggle_fallback and kaggle_dataset_slug:
        return _try_kaggle_download(kaggle_dataset_slug)

    return root


@dataclass
class _Source:
    name: str
    length: int

    def get_image(self, index: int) -> torch.Tensor:
        raise NotImplementedError

    def close(self) -> None:
        return None


class _ImageFolderSource(_Source):
    def __init__(self, name: str, root: Path) -> None:
        self._paths = _collect_image_files(root)
        super().__init__(name=name, length=len(self._paths))

    def get_image(self, index: int) -> torch.Tensor:
        return _load_image_tensor(self._paths[index])


class _LIU4KSource(_Source):
    def __init__(
        self,
        *,
        name: str,
        root: Path,
        split: str | None,
        auto_download_if_empty: bool,
    ) -> None:
        liu4k_dataset_class = _load_liu4k_dataset_class()
        self._dataset = liu4k_dataset_class(
            root=root,
            split=split,
            auto_download_if_empty=auto_download_if_empty,
            return_class=False,
            max_nested_zip_depth=0,
        )
        super().__init__(name=name, length=len(self._dataset))

    def get_image(self, index: int) -> torch.Tensor:
        return self._dataset[index]

    def close(self) -> None:
        self._dataset.close()


class _MNISTCsvSource(_Source):
    def __init__(self, name: str, root: Path) -> None:
        csv_candidates = list(root.rglob("*mnist*.csv")) + list(root.rglob("*.csv"))
        csv_path = next((p for p in csv_candidates if p.is_file()), None)
        if csv_path is None:
            raise RuntimeError(f"No MNIST csv file found in: {root}")

        with csv_path.open("r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)

        has_label = len(header) == 785 and "label" in header[0].lower()
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.uint16)
        if data.ndim == 1:
            data = data[None, :]

        pixels = data[:, 1:] if has_label else data
        if pixels.shape[1] != 28 * 28:
            raise RuntimeError(f"MNIST csv has unexpected width: {pixels.shape[1]}")

        self._images = pixels.reshape(-1, 28, 28).astype(np.float32) / 255.0
        super().__init__(name=name, length=self._images.shape[0])

    def get_image(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self._images[index])


class _MNISTIdxSource(_Source):
    def __init__(self, name: str, root: Path) -> None:
        idx_candidates = sorted(
            [p for p in root.rglob("*images*idx3-ubyte") if p.is_file()]
        )
        if not idx_candidates:
            raise RuntimeError(f"No MNIST idx image files found in: {root}")

        preferred = [p for p in idx_candidates if "train" in p.name.lower()]
        idx_path = preferred[0] if preferred else idx_candidates[0]

        with idx_path.open("rb") as f:
            header = f.read(16)
            if len(header) != 16:
                raise RuntimeError(f"MNIST idx file too short: {idx_path}")

            magic, count, rows, cols = struct.unpack(">IIII", header)
            if magic != 2051:
                raise RuntimeError(
                    f"Unexpected MNIST idx magic ({magic}) in: {idx_path}"
                )

            raw = f.read()

        expected = count * rows * cols
        data = np.frombuffer(raw, dtype=np.uint8)
        if data.size != expected:
            raise RuntimeError(
                f"MNIST idx payload mismatch in {idx_path}: expected {expected}, got {data.size}"
            )

        self._images = data.reshape(count, rows, cols).astype(np.float32) / 255.0
        super().__init__(name=name, length=self._images.shape[0])

    def get_image(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self._images[index])


class _CIFAR10BatchSource(_Source):
    def __init__(self, name: str, root: Path) -> None:
        batch_root = root
        if (root / "cifar-10-batches-py").exists():
            batch_root = root / "cifar-10-batches-py"

        batch_files = sorted(batch_root.glob("data_batch_*"))
        if not batch_files:
            raise RuntimeError(f"No CIFAR10 batch files found in: {root}")

        images = []
        for batch_path in batch_files:
            with batch_path.open("rb") as f:
                batch = pickle.load(f, encoding="latin1")
            data = batch.get("data")
            if data is None:
                data = batch.get(b"data")
            if data is None:
                continue

            rgb = data.reshape(-1, 3, 32, 32).astype(np.float32)
            gray = 0.299 * rgb[:, 0] + 0.587 * rgb[:, 1] + 0.114 * rgb[:, 2]
            images.append(gray / 255.0)

        if not images:
            raise RuntimeError(f"No image data decoded from CIFAR10 in: {root}")

        self._images = np.concatenate(images, axis=0)
        super().__init__(name=name, length=self._images.shape[0])

    def get_image(self, index: int) -> torch.Tensor:
        return torch.from_numpy(self._images[index])


def _random_patch(image: torch.Tensor, patch_size: int) -> torch.Tensor:
    if image.ndim != 2:
        raise ValueError(f"Expected 2D grayscale tensor, got shape: {tuple(image.shape)}")

    image = image.to(torch.float32)
    height, width = image.shape

    if height < patch_size or width < patch_size:
        scale = max(patch_size / height, patch_size / width)
        new_h = int(np.ceil(height * scale))
        new_w = int(np.ceil(width * scale))
        image = F.interpolate(
            image.unsqueeze(0).unsqueeze(0),
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0).squeeze(0)
        height, width = image.shape

    top = random.randint(0, height - patch_size)
    left = random.randint(0, width - patch_size)
    patch = image[top : top + patch_size, left : left + patch_size]
    return patch.unsqueeze(0)


class MixedCompressionDataset(Dataset):
    def __init__(
        self,
        *,
        sources: list[_Source],
        source_weights: list[float],
        patch_size: int = 128,
        epoch_length: int = 40000,
        return_source_name: bool = False,
    ) -> None:
        if patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if epoch_length <= 0:
            raise ValueError("epoch_length must be positive")
        if len(sources) != len(source_weights):
            raise ValueError("sources and source_weights must have the same length")

        active_pairs = [
            (source, float(weight))
            for source, weight in zip(sources, source_weights)
            if source.length > 0 and weight > 0
        ]
        if not active_pairs:
            raise RuntimeError("No active sources with positive weight and non-zero length")

        self.sources = [pair[0] for pair in active_pairs]
        self.weights = [pair[1] for pair in active_pairs]
        total_weight = sum(self.weights)
        self.probabilities = [w / total_weight for w in self.weights]
        self.patch_size = patch_size
        self.epoch_length = epoch_length
        self.return_source_name = return_source_name

    def __len__(self) -> int:
        return self.epoch_length

    def __getitem__(self, index: int):
        source = random.choices(self.sources, weights=self.probabilities, k=1)[0]
        sample_idx = random.randrange(source.length)
        image = source.get_image(sample_idx)
        patch = _random_patch(image, self.patch_size)

        if self.return_source_name:
            return patch, source.name
        return patch

    def close(self) -> None:
        for source in self.sources:
            source.close()


def build_mixed_compression_dataloader(
    *,
    liu4k_root: str | Path | None = None,
    mnist_root: str | Path | None = None,
    bsd_root: str | Path | None = None,
    cifar10_root: str | Path | None = None,
    liu4k_split: str = "train",
    patch_size: int = 128,
    epoch_length: int = 40000,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    auto_extract_archives: bool = True,
    enable_kaggle_fallback: bool = False,
    liu4k_auto_download_if_empty: bool = True,
    mnist_kaggle_slug: str = "hojjatk/mnist-dataset",
    bsd_kaggle_slug: str = "balraj98/berkeley-segmentation-dataset-500-bsds500",
    source_weights: dict[str, float] | None = None,
    return_source_name: bool = False,
) -> tuple[MixedCompressionDataset, DataLoader]:
    weights = source_weights or {
        "liu4k": 0.60,
        "mnist": 0.15,
        "bsd": 0.20,
        "cifar10": 0.05,
    }

    sources: list[_Source] = []
    source_probs: list[float] = []

    if liu4k_root is not None and weights.get("liu4k", 0) > 0:
        sources.append(
            _LIU4KSource(
                name="liu4k",
                root=Path(liu4k_root),
                split=liu4k_split,
                auto_download_if_empty=liu4k_auto_download_if_empty,
            )
        )
        source_probs.append(weights["liu4k"])

    if mnist_root is not None and weights.get("mnist", 0) > 0:
        prepared_root = _prepare_dataset_root(
            Path(mnist_root),
            auto_extract_archives=auto_extract_archives,
            kaggle_dataset_slug=mnist_kaggle_slug,
            enable_kaggle_fallback=enable_kaggle_fallback,
        )

        mnist_source: _Source
        mnist_idx_candidates = [
            p for p in prepared_root.rglob("*images*idx3-ubyte") if p.is_file()
        ]
        mnist_csv_candidates = [
            p for p in (list(prepared_root.rglob("*mnist*.csv")) + list(prepared_root.rglob("*.csv"))) if p.is_file()
        ]
        if mnist_idx_candidates:
            mnist_source = _MNISTIdxSource(name="mnist", root=prepared_root)
        elif mnist_csv_candidates:
            mnist_source = _MNISTCsvSource(name="mnist", root=prepared_root)
        else:
            mnist_source = _ImageFolderSource(name="mnist", root=prepared_root)

        sources.append(mnist_source)
        source_probs.append(weights["mnist"])

    if bsd_root is not None and weights.get("bsd", 0) > 0:
        prepared_root = _prepare_dataset_root(
            Path(bsd_root),
            auto_extract_archives=auto_extract_archives,
            kaggle_dataset_slug=bsd_kaggle_slug,
            enable_kaggle_fallback=enable_kaggle_fallback,
        )
        sources.append(_ImageFolderSource(name="bsd", root=prepared_root))
        source_probs.append(weights["bsd"])

    if cifar10_root is not None and weights.get("cifar10", 0) > 0:
        prepared_root = _prepare_dataset_root(
            Path(cifar10_root),
            auto_extract_archives=auto_extract_archives,
            kaggle_dataset_slug=None,
            enable_kaggle_fallback=False,
        )

        cifar_source: _Source
        batch_root = prepared_root / "cifar-10-batches-py"
        if batch_root.exists() or list(prepared_root.glob("data_batch_*")):
            cifar_source = _CIFAR10BatchSource(name="cifar10", root=prepared_root)
        else:
            cifar_source = _ImageFolderSource(name="cifar10", root=prepared_root)

        sources.append(cifar_source)
        source_probs.append(weights["cifar10"])

    dataset = MixedCompressionDataset(
        sources=sources,
        source_weights=source_probs,
        patch_size=patch_size,
        epoch_length=epoch_length,
        return_source_name=return_source_name,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataset, dataloader