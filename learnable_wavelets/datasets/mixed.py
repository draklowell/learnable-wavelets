import hashlib
import importlib.util
import io
import tarfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset

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
    return sorted([p for p in root.rglob("*") if p.is_file() and _is_image_file(p)])


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

    archives = [p for p in root.iterdir() if p.is_file() and _is_supported_archive(p)]

    for archive_path in archives:
        if archive_path.suffix.lower() == ".zip":
            with zipfile.ZipFile(archive_path, "r") as zf:
                zf.extractall(root)
            continue

        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(root)


def _try_kaggle_download(dataset_slug: str, retries: int = 3, retry_delay_sec: float = 2.0) -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "Kaggle fallback requires `kagglehub`. Install it with: pip install kagglehub"
        ) from exc

    last_error = None
    for attempt in range(1, retries + 1):
        try:
            return Path(kagglehub.dataset_download(dataset_slug))
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_delay_sec)

    raise RuntimeError(f"Kaggle download failed for {dataset_slug}") from last_error


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


def _stable_train_membership(key: str, split_seed: int, train_ratio: float) -> bool:
    digest = hashlib.md5(f"{split_seed}:{key}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return value < train_ratio


def _build_patch_transform(
    *,
    split: str,
    patch_size: int,
    normalize_mean: float | None = 0.5,
    normalize_std: float | None = 0.5,
):
    try:
        from torchvision import transforms as T
    except ImportError as exc:
        raise RuntimeError(
            "Patch transforms require torchvision. Install it with: pip install torchvision"
        ) from exc

    if split == "train":
        crop = T.RandomCrop(patch_size, pad_if_needed=True)
    else:
        crop = T.CenterCrop(patch_size)

    ops = [
        crop,
        T.Grayscale(num_output_channels=1),
        T.ToTensor(),
    ]
    if normalize_mean is not None and normalize_std is not None:
        ops.append(T.Normalize((normalize_mean,), (normalize_std,)))

    return T.Compose(ops)


def _default_dataset_root(name: str) -> Path:
    return Path(".datasets") / name


@dataclass
class _DatasetSource:
    name: str

    def __len__(self) -> int:
        raise NotImplementedError

    def key_at(self, index: int) -> str:
        raise NotImplementedError

    def pil_image_at(self, index: int) -> Image.Image:
        raise NotImplementedError

    def close(self) -> None:
        return None


class _ImageFolderSource(_DatasetSource):
    def __init__(self, name: str, root: Path) -> None:
        self.name = name
        self._paths = _collect_image_files(root)

    def __len__(self) -> int:
        return len(self._paths)

    def key_at(self, index: int) -> str:
        return str(self._paths[index].resolve())

    def pil_image_at(self, index: int) -> Image.Image:
        with Image.open(self._paths[index]) as img:
            return img.convert("RGB")


class _LIU4KImageSource(_DatasetSource):
    def __init__(
        self,
        *,
        name: str,
        root: Path,
        split: str,
        auto_download_if_empty: bool,
    ) -> None:
        self.name = name
        liu4k_dataset_class = _load_liu4k_dataset_class()
        self._dataset = liu4k_dataset_class(
            root=root,
            split=split,
            auto_download_if_empty=auto_download_if_empty,
            return_class=False,
            max_nested_zip_depth=0,
        )
        self._samples = self._dataset.samples

    def __len__(self) -> int:
        return len(self._samples)

    def key_at(self, index: int) -> str:
        source_type, source_ref, _ = self._samples[index]
        return f"{source_type}:{source_ref}"

    def pil_image_at(self, index: int) -> Image.Image:
        source_type, source_ref, _ = self._samples[index]
        if source_type == "file":
            with Image.open(source_ref) as img:
                return img.convert("RGB")

        zip_path_str, chain_str = source_ref.split("::", 1)
        zip_path = Path(zip_path_str)
        zf = self._dataset._get_zip(zip_path)

        current_zip = zf
        nested_zip_handles: list[zipfile.ZipFile] = []
        raw = None
        try:
            for i, member_name in enumerate(chain_str.split("||")):
                is_last = i == len(chain_str.split("||")) - 1
                with current_zip.open(member_name, "r") as f:
                    data = f.read()
                if is_last:
                    raw = data
                    break
                nested_zip = zipfile.ZipFile(io.BytesIO(data), "r")
                nested_zip_handles.append(nested_zip)
                current_zip = nested_zip
        finally:
            for handle in nested_zip_handles:
                handle.close()

        if raw is None:
            raise RuntimeError(f"Failed to read LIU4K sample at index {index}")

        with Image.open(io.BytesIO(raw)) as img:
            return img.convert("RGB")

    def close(self) -> None:
        self._dataset.close()


class MixedImageVisionDataset(VisionDataset):
    def __init__(
        self,
        *,
        split: str = "train",
        train_ratio: float = 0.9,
        split_seed: int = 42,
        transform=None,
        target_transform=None,
        liu4k_root: str | Path | None = None,
        include_liu4k: bool = True,
        liu4k_download_split: str = "train",
        liu4k_auto_download_if_empty: bool = True,
        coco_root: str | Path | None = None,
        include_coco: bool = False,
        div2k_root: str | Path | None = None,
        include_div2k: bool = True,
        bsd_root: str | Path | None = None,
        include_bsd: bool = True,
        auto_extract_archives: bool = True,
        enable_kaggle_fallback: bool = True,
        coco_kaggle_slug: str = "awsaf49/coco-2017-dataset",
        div2k_kaggle_slug: str = "soumikrakshit/div2k-high-resolution-images",
        bsd_kaggle_slug: str = "balraj98/berkeley-segmentation-dataset-500-bsds500",
        return_source_name: bool = False,
    ) -> None:
        super().__init__(root=".", transform=transform, target_transform=target_transform)

        split = split.lower()
        if split not in {"train", "valid", "validation", "all"}:
            raise ValueError("split must be one of: train, valid, validation, all")
        if not (0 < train_ratio < 1):
            raise ValueError("train_ratio must be between 0 and 1")

        self.split = "valid" if split == "validation" else split
        self.train_ratio = train_ratio
        self.split_seed = split_seed
        self.return_source_name = return_source_name

        self._sources: list[_DatasetSource] = []

        if include_liu4k:
            resolved_liu4k_root = (
                Path(liu4k_root) if liu4k_root is not None else Path("liu4k") / liu4k_download_split
            )
            self._sources.append(
                _LIU4KImageSource(
                    name="liu4k",
                    root=resolved_liu4k_root,
                    split=liu4k_download_split,
                    auto_download_if_empty=liu4k_auto_download_if_empty,
                )
            )

        if include_coco:
            coco_prepared = _prepare_dataset_root(
                Path(coco_root) if coco_root is not None else _default_dataset_root("coco"),
                auto_extract_archives=auto_extract_archives,
                kaggle_dataset_slug=coco_kaggle_slug,
                enable_kaggle_fallback=enable_kaggle_fallback,
            )
            self._sources.append(_ImageFolderSource(name="coco", root=coco_prepared))

        if include_div2k:
            div2k_prepared = _prepare_dataset_root(
                Path(div2k_root) if div2k_root is not None else _default_dataset_root("div2k"),
                auto_extract_archives=auto_extract_archives,
                kaggle_dataset_slug=div2k_kaggle_slug,
                enable_kaggle_fallback=enable_kaggle_fallback,
            )
            self._sources.append(_ImageFolderSource(name="div2k", root=div2k_prepared))

        if include_bsd:
            bsd_prepared = _prepare_dataset_root(
                Path(bsd_root) if bsd_root is not None else _default_dataset_root("bsd"),
                auto_extract_archives=auto_extract_archives,
                kaggle_dataset_slug=bsd_kaggle_slug,
                enableqq_kaggle_fallback=enable_kaggle_fallback,
            )
            self._sources.append(_ImageFolderSource(name="bsd", root=bsd_prepared))

        self._samples: list[tuple[str, _DatasetSource, int]] = []
        for source in self._sources:
            for index in range(len(source)):
                key = f"{source.name}::{source.key_at(index)}"
                is_train = _stable_train_membership(key, split_seed, train_ratio)

                if self.split == "all":
                    self._samples.append((source.name, source, index))
                elif self.split == "train" and is_train:
                    self._samples.append((source.name, source, index))
                elif self.split == "valid" and not is_train:
                    self._samples.append((source.name, source, index))

        if not self._samples:
            raise RuntimeError("No images available for the selected split and sources")

        self.active_sources = [(source.name, len(source)) for source in self._sources if len(source) > 0]

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int):
        source_name, source, source_index = self._samples[index]
        image = source.pil_image_at(source_index)
        target = source_name if self.return_source_name else 0

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def close(self) -> None:
        for source in self._sources:
            source.close()


def build_mixed_vision_dataloader(
    *,
    split: str = "train",
    train_ratio: float = 0.9,
    split_seed: int = 42,
    patch_size: int = 128,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool = False,
    normalize_mean: float | None = 0.5,
    normalize_std: float | None = 0.5,
    liu4k_root: str | Path | None = None,
    include_liu4k: bool = True,
    liu4k_download_split: str = "train",
    liu4k_auto_download_if_empty: bool = True,
    coco_root: str | Path | None = None,
    include_coco: bool = False,
    div2k_root: str | Path | None = None,
    include_div2k: bool = True,
    bsd_root: str | Path | None = None,
    include_bsd: bool = True,
    auto_extract_archives: bool = True,
    enable_kaggle_fallback: bool = True,
    coco_kaggle_slug: str = "awsaf49/coco-2017-dataset",
    div2k_kaggle_slug: str = "soumikrakshit/div2k-high-resolution-images",
    bsd_kaggle_slug: str = "balraj98/berkeley-segmentation-dataset-500-bsds500",
    return_source_name: bool = False,
) -> tuple[MixedImageVisionDataset, DataLoader]:
    transform = _build_patch_transform(
        split="train" if split == "train" else "valid",
        patch_size=patch_size,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
    )

    dataset = MixedImageVisionDataset(
        split=split,
        train_ratio=train_ratio,
        split_seed=split_seed,
        transform=transform,
        liu4k_root=liu4k_root,
        include_liu4k=include_liu4k,
        liu4k_download_split=liu4k_download_split,
        liu4k_auto_download_if_empty=liu4k_auto_download_if_empty,
        coco_root=coco_root,
        include_coco=include_coco,
        div2k_root=div2k_root,
        include_div2k=include_div2k,
        bsd_root=bsd_root,
        include_bsd=include_bsd,
        auto_extract_archives=auto_extract_archives,
        enable_kaggle_fallback=enable_kaggle_fallback,
        coco_kaggle_slug=coco_kaggle_slug,
        div2k_kaggle_slug=div2k_kaggle_slug,
        bsd_kaggle_slug=bsd_kaggle_slug,
        return_source_name=return_source_name,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=split == "train",
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return dataset, dataloader
