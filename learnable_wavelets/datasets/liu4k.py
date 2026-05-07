import argparse
import io
import json
import os
import random
import re
import threading
import zipfile
from bisect import bisect_right
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable
from zipfile import BadZipFile, ZipFile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_ZIP_PART_PATTERN = re.compile(r"\.z(\d+)$", re.IGNORECASE)
_ZIPFILE_PATCH_LOCK = threading.Lock()
DEFAULT_PATCH_SIZE = 200
DEFAULT_SOURCE_FRACTION = 1.0 / 3.0
DEFAULT_TRAIN_COUNT = 160_000
DEFAULT_VALIDATION_COUNT = 20_000
DEFAULT_SEED = 42


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _is_split_zip_part(path: Path) -> bool:
    return SPLIT_ZIP_PART_PATTERN.fullmatch(path.suffix.lower()) is not None


def _is_zip_file(path: Path) -> bool:
    return path.suffix.lower() == ".zip"


def _is_archive_file(path: Path) -> bool:
    return _is_zip_file(path) or _is_split_zip_part(path)


def _archive_group_key(path: Path) -> str:
    if not _is_archive_file(path):
        raise ValueError(f"Not a zip archive part: {path}")
    return path.with_suffix("").as_posix()


def _archive_part_sort_key(path: Path) -> tuple[int, int | str]:
    split_match = SPLIT_ZIP_PART_PATTERN.fullmatch(path.suffix.lower())
    if split_match is not None:
        return (0, int(split_match.group(1)))
    if _is_zip_file(path):
        return (1, path.name)
    return (2, path.name)


def _collect_image_files(root: Path) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"LIU4K root does not exist: {root}")

    files = sorted(p for p in root.rglob("*") if p.is_file() and _is_image_file(p))
    if not files:
        raise RuntimeError(f"No images found under LIU4K root: {root}")
    return files


def _sample_uniform_third(
    image_paths: list[Path],
    *,
    source_fraction: float,
    seed: int,
) -> list[Path]:
    if not (0 < source_fraction <= 1):
        raise ValueError("source_fraction must be in the range (0, 1]")

    sample_size = max(1, int(len(image_paths) * source_fraction))
    rng = random.Random(seed)
    return sorted(rng.sample(image_paths, sample_size))


def _iter_patch_boxes(
    width: int, height: int, patch_size: int
) -> Iterable[tuple[int, int]]:
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")

    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            yield x, y


def _image_patch_refs(
    image_paths: Iterable[Path],
    *,
    patch_size: int,
) -> list[tuple[Path, int, int]]:
    refs: list[tuple[Path, int, int]] = []

    for path in image_paths:
        with Image.open(path) as image:
            width, height = image.size

        for x, y in _iter_patch_boxes(width, height, patch_size):
            refs.append((path, x, y))

    return refs


def _split_patch_refs(
    patch_refs: list[tuple[Path, int, int]],
    *,
    train_count: int,
    validation_count: int,
    seed: int,
) -> tuple[list[tuple[Path, int, int]], list[tuple[Path, int, int]]]:
    if train_count < 0 or validation_count < 0:
        raise ValueError("train_count and validation_count must be non-negative")

    required_count = train_count + validation_count
    if len(patch_refs) < required_count:
        raise RuntimeError(
            "The sampled LIU4K images produced "
            f"{len(patch_refs)} non-overlapping patches, but {required_count} "
            "are required. Increase source_fraction or reduce the split sizes."
        )

    rng = random.Random(seed)
    shuffled = patch_refs.copy()
    rng.shuffle(shuffled)

    train_refs = shuffled[:train_count]
    validation_refs = shuffled[train_count:required_count]
    return train_refs, validation_refs


def _prepare_output_split(split_dir: Path, *, overwrite: bool) -> None:
    if split_dir.exists() and any(split_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output split already contains files: {split_dir}. "
            "Pass overwrite=True or choose an empty output directory."
        )

    split_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in split_dir.glob("*.png"):
            path.unlink()


def _save_png_patch(
    *,
    image: Image.Image,
    output_path: Path,
    x: int,
    y: int,
    patch_size: int,
    png_compress_level: int,
) -> None:
    patch = image.crop((x, y, x + patch_size, y + patch_size))
    patch.save(
        output_path,
        format="PNG",
        compress_level=png_compress_level,
    )


def _write_split(
    refs: list[tuple[Path, int, int]],
    *,
    split_dir: Path,
    patch_size: int,
    png_compress_level: int,
) -> None:
    current_image_path: Path | None = None
    current_image: Image.Image | None = None
    ordered_refs = sorted(
        enumerate(refs),
        key=lambda item: (item[1][0], item[1][2], item[1][1]),
    )

    try:
        for index, (image_path, x, y) in ordered_refs:
            if image_path != current_image_path:
                if current_image is not None:
                    current_image.close()
                current_image = Image.open(image_path)
                current_image.load()
                current_image_path = image_path

            output_path = split_dir / f"{index:06d}.png"
            _save_png_patch(
                image=current_image,
                output_path=output_path,
                x=x,
                y=y,
                patch_size=patch_size,
                png_compress_level=png_compress_level,
            )
    finally:
        if current_image is not None:
            current_image.close()


@contextmanager
def _allow_multidisk_zipfile():
    original = zipfile._EndRecData64

    def _patched_endrecdata64(fpin, offset, endrec):
        try:
            return original(fpin, offset, endrec)
        except BadZipFile as exc:
            if "span multiple disks" not in str(exc):
                raise

            try:
                fpin.seek(offset - zipfile.sizeEndCentDir64Locator, io.SEEK_END)
            except OSError:
                return endrec

            data = fpin.read(zipfile.sizeEndCentDir64Locator)
            if len(data) != zipfile.sizeEndCentDir64Locator:
                return endrec

            sig, _, reloff, _ = zipfile.struct.unpack(
                zipfile.structEndArchive64Locator, data
            )
            if sig != zipfile.stringEndArchive64Locator:
                return endrec

            fpin.seek(
                offset - zipfile.sizeEndCentDir64Locator - zipfile.sizeEndCentDir64,
                io.SEEK_END,
            )
            data = fpin.read(zipfile.sizeEndCentDir64)
            if len(data) != zipfile.sizeEndCentDir64:
                return endrec

            (
                sig,
                _sz,
                _create_version,
                _read_version,
                disk_num,
                disk_dir,
                _dircount,
                dircount2,
                dirsize,
                diroffset,
            ) = zipfile.struct.unpack(zipfile.structEndArchive64, data)
            if sig != zipfile.stringEndArchive64:
                return endrec

            if disk_num != 0 or disk_dir != 0 or reloff >= 0:
                diroffset += reloff

            endrec[zipfile._ECD_SIGNATURE] = sig
            endrec[zipfile._ECD_DISK_NUMBER] = 0
            endrec[zipfile._ECD_DISK_START] = 0
            endrec[zipfile._ECD_ENTRIES_THIS_DISK] = dircount2
            endrec[zipfile._ECD_ENTRIES_TOTAL] = dircount2
            endrec[zipfile._ECD_SIZE] = dirsize
            endrec[zipfile._ECD_OFFSET] = diroffset
            return endrec

    with _ZIPFILE_PATCH_LOCK:
        zipfile._EndRecData64 = _patched_endrecdata64
        try:
            yield
        finally:
            zipfile._EndRecData64 = original


class _SplitZipStream(io.BufferedIOBase):
    def __init__(self, parts: Iterable[Path]) -> None:
        super().__init__()
        self._parts = tuple(parts)
        if not self._parts:
            raise ValueError("Split zip stream requires at least one part")

        self._files = [part.open("rb") for part in self._parts]
        self._sizes = [part.stat().st_size for part in self._parts]
        self._offsets: list[int] = []
        current = 0
        for size in self._sizes:
            self._offsets.append(current)
            current += size
        self._total_size = current
        self._position = 0

    def _locate_part(self, absolute_offset: int) -> int:
        if absolute_offset >= self._total_size:
            return len(self._offsets) - 1
        return max(0, bisect_right(self._offsets, absolute_offset) - 1)

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        if self.closed:
            raise ValueError("I/O operation on closed file")
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if self.closed:
            raise ValueError("I/O operation on closed file")

        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self._position + offset
        elif whence == io.SEEK_END:
            target = self._total_size + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")

        if target < 0:
            raise ValueError("Negative seek position")

        self._position = min(target, self._total_size)
        return self._position

    def read(self, size: int = -1) -> bytes:
        if self.closed:
            raise ValueError("I/O operation on closed file")

        if size is None or size < 0:
            size = self._total_size - self._position

        if size == 0 or self._position >= self._total_size:
            return b""

        remaining = min(size, self._total_size - self._position)
        chunks: list[bytes] = []

        while remaining > 0:
            part_idx = self._locate_part(self._position)
            part_start = self._offsets[part_idx]
            offset_inside_part = self._position - part_start
            available_in_part = self._sizes[part_idx] - offset_inside_part
            to_read = min(remaining, available_in_part)

            file = self._files[part_idx]
            file.seek(offset_inside_part)
            chunk = file.read(to_read)
            if not chunk:
                break

            chunks.append(chunk)
            read_len = len(chunk)
            self._position += read_len
            remaining -= read_len

            if read_len < to_read:
                break

        return b"".join(chunks)

    def close(self) -> None:
        if self.closed:
            return
        for file in self._files:
            file.close()
        super().close()


def _part_offsets(parts: Iterable[Path]) -> list[int]:
    offsets: list[int] = []
    running = 0
    for part in parts:
        offsets.append(running)
        running += part.stat().st_size
    return offsets


def _has_local_file_header(zf: ZipFile, offset: int) -> bool:
    if offset < 0:
        return False

    current_pos = zf.fp.tell()
    try:
        zf.fp.seek(offset, io.SEEK_SET)
        return zf.fp.read(4) == b"PK\x03\x04"
    finally:
        zf.fp.seek(current_pos, io.SEEK_SET)


def _patch_split_member_offsets(zf: ZipFile, parts: tuple[Path, ...]) -> None:
    disk_offsets = _part_offsets(parts)
    first_raw_offset = min((info.header_offset for info in zf.filelist), default=0)

    current_pos = zf.fp.tell()
    try:
        zf.fp.seek(0, io.SEEK_SET)
        prefix_probe = zf.fp.read(128)
    finally:
        zf.fp.seek(current_pos, io.SEEK_SET)

    split_prefix_shift = prefix_probe.find(b"PK\x03\x04")
    if split_prefix_shift < 0:
        split_prefix_shift = 0

    for info in zf.filelist:
        volume = getattr(info, "volume", 0)
        volume_shift = disk_offsets[volume] if 0 <= volume < len(disk_offsets) else 0
        current_offset = info.header_offset
        candidates = (
            current_offset,
            current_offset + volume_shift,
            current_offset - first_raw_offset + split_prefix_shift,
            current_offset - first_raw_offset + volume_shift + split_prefix_shift,
        )

        for candidate_offset in dict.fromkeys(candidates):
            if _has_local_file_header(zf, candidate_offset):
                info.header_offset = candidate_offset
                break


@contextmanager
def _open_archive_group_zip(parts: tuple[Path, ...]):
    if len(parts) == 1:
        with ZipFile(parts[0], "r") as zf:
            yield zf
        return

    stream = _SplitZipStream(parts)
    try:
        with _allow_multidisk_zipfile():
            with ZipFile(stream, "r") as zf:
                _patch_split_member_offsets(zf, parts)
                yield zf
    finally:
        stream.close()


@dataclass(frozen=True)
class ArchiveGroup:
    key: str
    parts: tuple[Path, ...]


@dataclass(frozen=True)
class LIU4KBuildManifest:
    raw_root: str
    output_root: str
    patch_size: int
    source_fraction: float
    seed: int
    selected_source_images: int
    available_patches: int
    train_count: int
    validation_count: int
    source_kind: str = "images"
    selected_archive_groups: int = 0
    compression: str = "PNG lossless"


@dataclass(frozen=True)
class LIU4KDownloadManifest:
    source: str
    destination_root: str
    listed_files: int
    selected_files: int
    downloaded_files: int
    selected_archive_groups: int = 0
    mode: str = "files"


def _download_gdrive_file(
    *,
    destination_dir: Path,
    gdrive_url: str | None = None,
    gdrive_file_id: str | None = None,
    quiet: bool = False,
    resume: bool = True,
) -> Path:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive downloads require `gdown`. Install it with: pip install gdown"
        ) from exc

    destination_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = gdown.download(
        url=gdrive_url,
        id=gdrive_file_id,
        output=str(destination_dir) + os.sep,
        quiet=quiet,
        resume=resume,
    )
    if downloaded_path is None:
        raise RuntimeError("Google Drive file download failed")
    return Path(downloaded_path)


def _list_gdrive_folder_files(
    *,
    gdrive_folder_url: str | None = None,
    gdrive_folder_id: str | None = None,
    destination_dir: Path,
    quiet: bool = False,
):
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive downloads require `gdown`. Install it with: pip install gdown"
        ) from exc

    files = gdown.download_folder(
        url=gdrive_folder_url,
        id=gdrive_folder_id,
        output=str(destination_dir),
        quiet=quiet,
        skip_download=True,
    )
    return sorted(files, key=lambda item: item.path)


def _archive_groups_from_paths(paths: Iterable[Path]) -> list[ArchiveGroup]:
    groups: dict[str, list[Path]] = {}
    for path in paths:
        if not _is_archive_file(path):
            continue
        groups.setdefault(_archive_group_key(path), []).append(path)

    archive_groups: list[ArchiveGroup] = []
    for key, parts in groups.items():
        sorted_parts = tuple(sorted(parts, key=_archive_part_sort_key))
        has_zip = any(_is_zip_file(part) for part in sorted_parts)
        if not has_zip:
            continue
        archive_groups.append(ArchiveGroup(key=key, parts=sorted_parts))

    return sorted(archive_groups, key=lambda group: group.key)


def _collect_archive_groups(root: Path) -> list[ArchiveGroup]:
    if not root.exists():
        raise FileNotFoundError(f"LIU4K archive root does not exist: {root}")

    groups = _archive_groups_from_paths(
        path for path in root.rglob("*") if path.is_file()
    )
    if not groups:
        raise RuntimeError(f"No zip archives or split zip groups found under: {root}")
    return groups


def _archive_groups_from_gdrive_items(items) -> dict[str, list]:
    groups: dict[str, list] = {}
    for item in items:
        item_path = Path(item.path)
        if not _is_archive_file(item_path):
            continue
        groups.setdefault(_archive_group_key(item_path), []).append(item)

    complete_groups: dict[str, list] = {}
    for key, group_items in groups.items():
        sorted_items = sorted(
            group_items,
            key=lambda item: _archive_part_sort_key(Path(item.path)),
        )
        if any(_is_zip_file(Path(item.path)) for item in sorted_items):
            complete_groups[key] = sorted_items

    return dict(sorted(complete_groups.items()))


def _sample_uniform_archive_groups(
    archive_groups: list[ArchiveGroup],
    *,
    source_fraction: float,
    seed: int,
) -> list[ArchiveGroup]:
    if not (0 < source_fraction <= 1):
        raise ValueError("source_fraction must be in the range (0, 1]")

    selected_count = max(1, int(len(archive_groups) * source_fraction))
    rng = random.Random(seed)
    selected = rng.sample(archive_groups, selected_count)
    rng.shuffle(selected)
    return selected


def _sample_gdrive_folder_files(
    files,
    *,
    source_fraction: float,
    seed: int,
) -> list:
    image_files = [item for item in files if _is_image_file(Path(item.path))]
    if not image_files:
        raise RuntimeError(
            "The Google Drive folder listing does not contain individual image files. "
            "Selective download cannot avoid the whole dataset when LIU4K is packaged "
            "as archives or split archives. Download/extract those files outside this "
            "builder, then pass the extracted folder with --raw-root."
        )

    selected_count = max(1, int(len(image_files) * source_fraction))
    rng = random.Random(seed)
    return sorted(rng.sample(image_files, selected_count), key=lambda item: item.path)


def _download_gdrive_item(item, *, quiet: bool, resume: bool) -> bool:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "Google Drive downloads require `gdown`. Install it with: pip install gdown"
        ) from exc

    local_path = Path(item.local_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    downloaded_path = gdown.download(
        id=item.id,
        output=str(local_path),
        quiet=quiet,
        resume=resume,
    )
    return downloaded_path is not None


def download_liu4k_gdrive_subset(
    destination_dir: str | Path,
    *,
    gdrive_folder_url: str | None = None,
    gdrive_folder_id: str | None = None,
    gdrive_file_url: str | None = None,
    gdrive_file_id: str | None = None,
    source_fraction: float = DEFAULT_SOURCE_FRACTION,
    seed: int = DEFAULT_SEED,
    quiet: bool = False,
    resume: bool = True,
) -> LIU4KDownloadManifest:
    """Download LIU4K source files from Google Drive.

    Folder downloads are selective: the folder is listed first, then a uniform
    fraction of complete split-zip groups is downloaded. If the folder contains
    individual images instead of archives, a uniform fraction of images is used.
    Direct file downloads are explicit and download only that single Drive file.
    """

    destination_dir = Path(destination_dir)
    folder_source_count = sum(
        source is not None for source in (gdrive_folder_url, gdrive_folder_id)
    )
    file_source_count = sum(
        source is not None for source in (gdrive_file_url, gdrive_file_id)
    )
    if folder_source_count + file_source_count != 1:
        raise ValueError(
            "Provide exactly one of gdrive_folder_url, gdrive_folder_id, "
            "gdrive_file_url, or gdrive_file_id."
        )
    if not (0 < source_fraction <= 1):
        raise ValueError("source_fraction must be in the range (0, 1]")

    if file_source_count:
        downloaded_path = _download_gdrive_file(
            destination_dir=destination_dir,
            gdrive_url=gdrive_file_url,
            gdrive_file_id=gdrive_file_id,
            quiet=quiet,
            resume=resume,
        )
        return LIU4KDownloadManifest(
            source=gdrive_file_url or f"file-id:{gdrive_file_id}",
            destination_root=str(destination_dir.resolve()),
            listed_files=1,
            selected_files=1,
            downloaded_files=1 if downloaded_path.exists() else 0,
            mode="single-file",
        )

    files = _list_gdrive_folder_files(
        gdrive_folder_url=gdrive_folder_url,
        gdrive_folder_id=gdrive_folder_id,
        destination_dir=destination_dir,
        quiet=quiet,
    )

    archive_groups = _archive_groups_from_gdrive_items(files)
    if archive_groups:
        group_items = list(archive_groups.items())
        selected_count = max(1, int(len(group_items) * source_fraction))
        rng = random.Random(seed)
        selected_groups = rng.sample(group_items, selected_count)
        rng.shuffle(selected_groups)
        selected_items = [
            item for _group_key, group_parts in selected_groups for item in group_parts
        ]

        downloaded_count = sum(
            1
            for item in selected_items
            if _download_gdrive_item(item, quiet=quiet, resume=resume)
        )
        return LIU4KDownloadManifest(
            source=gdrive_folder_url or f"folder-id:{gdrive_folder_id}",
            destination_root=str(destination_dir.resolve()),
            listed_files=len(files),
            selected_files=len(selected_items),
            downloaded_files=downloaded_count,
            selected_archive_groups=len(selected_groups),
            mode="archive-groups",
        )

    selected_files = _sample_gdrive_folder_files(
        files,
        source_fraction=source_fraction,
        seed=seed,
    )

    downloaded_count = sum(
        1
        for item in selected_files
        if _download_gdrive_item(item, quiet=quiet, resume=resume)
    )

    return LIU4KDownloadManifest(
        source=gdrive_folder_url or f"folder-id:{gdrive_folder_id}",
        destination_root=str(destination_dir.resolve()),
        listed_files=len(files),
        selected_files=len(selected_files),
        downloaded_files=downloaded_count,
        mode="image-files",
    )


def build_liu4k_patches(
    raw_root: str | Path,
    output_root: str | Path,
    *,
    patch_size: int = DEFAULT_PATCH_SIZE,
    source_fraction: float = DEFAULT_SOURCE_FRACTION,
    train_count: int = DEFAULT_TRAIN_COUNT,
    validation_count: int = DEFAULT_VALIDATION_COUNT,
    seed: int = DEFAULT_SEED,
    png_compress_level: int = 9,
    overwrite: bool = False,
) -> LIU4KBuildManifest:
    """Build a losslessly compressed LIU4K patch dataset.

    The builder samples source images uniformly without replacement, cuts only
    non-overlapping full patches, shuffles those patch references, and writes
    train/validation PNG files. PNG compression is lossless, so this pipeline
    does not add lossy compression beyond whatever format the raw data already
    used.
    """

    if not (0 <= png_compress_level <= 9):
        raise ValueError("png_compress_level must be between 0 and 9")

    raw_root = Path(raw_root)
    output_root = Path(output_root)
    train_dir = output_root / "train"
    validation_dir = output_root / "validation"

    image_paths = _collect_image_files(raw_root)
    selected_images = _sample_uniform_third(
        image_paths,
        source_fraction=source_fraction,
        seed=seed,
    )
    patch_refs = _image_patch_refs(selected_images, patch_size=patch_size)
    train_refs, validation_refs = _split_patch_refs(
        patch_refs,
        train_count=train_count,
        validation_count=validation_count,
        seed=seed,
    )

    _prepare_output_split(train_dir, overwrite=overwrite)
    _prepare_output_split(validation_dir, overwrite=overwrite)

    _write_split(
        train_refs,
        split_dir=train_dir,
        patch_size=patch_size,
        png_compress_level=png_compress_level,
    )
    _write_split(
        validation_refs,
        split_dir=validation_dir,
        patch_size=patch_size,
        png_compress_level=png_compress_level,
    )

    manifest = LIU4KBuildManifest(
        raw_root=str(raw_root.resolve()),
        output_root=str(output_root.resolve()),
        patch_size=patch_size,
        source_fraction=source_fraction,
        seed=seed,
        selected_source_images=len(selected_images),
        available_patches=len(patch_refs),
        train_count=len(train_refs),
        validation_count=len(validation_refs),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2)
        f.write("\n")

    return manifest


def build_liu4k_patches_from_archives(
    archive_root: str | Path,
    output_root: str | Path,
    *,
    patch_size: int = DEFAULT_PATCH_SIZE,
    source_fraction: float = DEFAULT_SOURCE_FRACTION,
    train_count: int = DEFAULT_TRAIN_COUNT,
    validation_count: int = DEFAULT_VALIDATION_COUNT,
    seed: int = DEFAULT_SEED,
    png_compress_level: int = 9,
    overwrite: bool = False,
    cleanup_archives: bool = False,
) -> LIU4KBuildManifest:
    """Build patches from local zip files or complete split-zip groups."""

    if not (0 <= png_compress_level <= 9):
        raise ValueError("png_compress_level must be between 0 and 9")

    archive_root = Path(archive_root)
    output_root = Path(output_root)
    train_dir = output_root / "train"
    validation_dir = output_root / "validation"
    required_count = train_count + validation_count

    archive_groups = _collect_archive_groups(archive_root)
    selected_groups = _sample_uniform_archive_groups(
        archive_groups,
        source_fraction=source_fraction,
        seed=seed,
    )

    _prepare_output_split(train_dir, overwrite=overwrite)
    _prepare_output_split(validation_dir, overwrite=overwrite)

    rng = random.Random(seed)
    target_splits = ["train"] * train_count + ["validation"] * validation_count
    rng.shuffle(target_splits)
    split_indices = {"train": 0, "validation": 0}
    written_patches = 0
    processed_images = 0

    for group in selected_groups:
        try:
            with _open_archive_group_zip(group.parts) as zf:
                image_infos = [
                    info
                    for info in zf.infolist()
                    if not info.is_dir() and _is_image_file(Path(info.filename))
                ]
                rng.shuffle(image_infos)

                for info in image_infos:
                    with zf.open(info, "r") as file:
                        data = file.read()

                    with Image.open(io.BytesIO(data)) as image:
                        image.load()
                        processed_images += 1
                        boxes = list(
                            _iter_patch_boxes(
                                image.width,
                                image.height,
                                patch_size,
                            )
                        )
                        rng.shuffle(boxes)

                        for x, y in boxes:
                            if written_patches >= required_count:
                                break

                            split = target_splits[written_patches]
                            split_dir = (
                                train_dir if split == "train" else validation_dir
                            )
                            output_path = split_dir / f"{split_indices[split]:06d}.png"
                            _save_png_patch(
                                image=image,
                                output_path=output_path,
                                x=x,
                                y=y,
                                patch_size=patch_size,
                                png_compress_level=png_compress_level,
                            )
                            split_indices[split] += 1
                            written_patches += 1

                    if written_patches >= required_count:
                        break
        finally:
            if cleanup_archives:
                for part in group.parts:
                    part.unlink(missing_ok=True)

        if written_patches >= required_count:
            break

    if written_patches < required_count:
        raise RuntimeError(
            "The selected LIU4K archive groups produced "
            f"{written_patches} patches, but {required_count} are required. "
            "Increase --download-fraction/--source-fraction or reduce split sizes."
        )

    manifest = LIU4KBuildManifest(
        raw_root=str(archive_root.resolve()),
        output_root=str(output_root.resolve()),
        patch_size=patch_size,
        source_fraction=source_fraction,
        seed=seed,
        selected_source_images=processed_images,
        available_patches=written_patches,
        train_count=split_indices["train"],
        validation_count=split_indices["validation"],
        source_kind="archive-groups",
        selected_archive_groups=len(selected_groups),
    )

    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(manifest), f, indent=2)
        f.write("\n")

    return manifest


def _pil_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image)

    if array.ndim == 2:
        array = array[:, :, None]

    if array.dtype == np.uint8:
        tensor = torch.from_numpy(array.copy()).permute(2, 0, 1).float()
        return tensor / 255.0

    tensor = torch.from_numpy(array.copy()).permute(2, 0, 1)
    return tensor.to(torch.float32)


def _normalize_split(split: str) -> str:
    normalized = split.strip().lower()
    aliases = {
        "all": "all",
        "train": "train",
        "training": "train",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "dev": "validation",
    }
    if normalized not in aliases:
        raise ValueError("split must be one of: train, validation, all")
    return aliases[normalized]


class LIU4KDataset(Dataset):
    """PyTorch dataset for prepared lossless LIU4K 200x200 PNG patches."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str = "train",
        transform=None,
        return_path: bool = False,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.split = _normalize_split(split)
        self.transform = transform
        self.return_path = return_path

        if self.split == "all":
            split_roots = [
                split_root
                for split_root in (self.root / "train", self.root / "validation")
                if split_root.exists()
            ]
            if not split_roots:
                split_roots = [self.root]
        else:
            split_root = self.root / self.split
            split_roots = [split_root if split_root.exists() else self.root]

        self.split_roots = split_roots
        self.paths = sorted(
            path
            for split_root in self.split_roots
            for path in split_root.rglob("*.png")
        )
        self.samples = self.paths

        if not self.paths:
            raise RuntimeError(
                f"No prepared LIU4K PNG patches found in {self.root}. "
                "Run build_liu4k_patches(...) first."
            )

    def __len__(self) -> int:
        return len(self.paths)

    def pil_image_at(self, index: int) -> Image.Image:
        with Image.open(self.paths[index]) as image:
            return image.copy()

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = self.pil_image_at(index)

        if self.transform is not None:
            sample = self.transform(image)
        else:
            sample = _pil_to_tensor(image)

        if self.return_path:
            return sample, str(path)
        return sample


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare LIU4K as lossless 200x200 PNG patches from a local or "
            "mounted source folder, or from a selective Google Drive download."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--raw-root",
        "--source-root",
        dest="raw_root",
        help="Path to local or mounted LIU4K images.",
    )
    source_group.add_argument(
        "--gdrive-folder-url",
        help=(
            "Google Drive folder URL. The folder is listed first and only a "
            "uniform fraction of complete split-zip groups is downloaded."
        ),
    )
    source_group.add_argument(
        "--gdrive-folder-id",
        help=(
            "Google Drive folder ID. The folder is listed first and only a "
            "uniform fraction of complete split-zip groups is downloaded."
        ),
    )
    source_group.add_argument(
        "--gdrive-file-url",
        help="Google Drive file URL. Downloads exactly this one file.",
    )
    source_group.add_argument(
        "--gdrive-file-id",
        help="Google Drive file ID. Downloads exactly this one file.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Destination folder for train/validation PNG patches.",
    )
    parser.add_argument(
        "--download-root",
        default=".datasets/liu4k-gdrive-source",
        help="Destination folder for files downloaded from Google Drive.",
    )
    parser.add_argument(
        "--download-fraction",
        type=float,
        default=None,
        help=(
            "Fraction of split-zip groups or image files to download from a "
            "Google Drive folder. Defaults to --source-fraction."
        ),
    )
    parser.add_argument(
        "--quiet-download",
        action="store_true",
        help="Suppress gdown download output.",
    )
    parser.add_argument(
        "--no-resume-download",
        action="store_true",
        help="Disable resuming partial Google Drive downloads.",
    )
    parser.add_argument("--patch-size", type=int, default=DEFAULT_PATCH_SIZE)
    parser.add_argument(
        "--source-fraction", type=float, default=DEFAULT_SOURCE_FRACTION
    )
    parser.add_argument("--train-count", type=int, default=DEFAULT_TRAIN_COUNT)
    parser.add_argument(
        "--validation-count",
        type=int,
        default=DEFAULT_VALIDATION_COUNT,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--png-compress-level", type=int, default=9)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--cleanup-archives-after-build",
        action="store_true",
        help="Delete selected local archive parts after patches are written.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()

    raw_root = args.raw_root
    source_fraction = args.source_fraction
    download_manifest = None

    if raw_root is None:
        download_fraction = (
            args.download_fraction
            if args.download_fraction is not None
            else args.source_fraction
        )
        download_manifest = download_liu4k_gdrive_subset(
            destination_dir=args.download_root,
            gdrive_folder_url=args.gdrive_folder_url,
            gdrive_folder_id=args.gdrive_folder_id,
            gdrive_file_url=args.gdrive_file_url,
            gdrive_file_id=args.gdrive_file_id,
            source_fraction=download_fraction,
            seed=args.seed,
            quiet=args.quiet_download,
            resume=not args.no_resume_download,
        )
        raw_root = args.download_root
        source_fraction = 1.0

    raw_root_path = Path(raw_root)
    local_archive_groups = (
        _archive_groups_from_paths(
            path for path in raw_root_path.rglob("*") if path.is_file()
        )
        if raw_root_path.exists()
        else []
    )
    if local_archive_groups:
        manifest = build_liu4k_patches_from_archives(
            archive_root=raw_root,
            output_root=args.output_root,
            patch_size=args.patch_size,
            source_fraction=source_fraction,
            train_count=args.train_count,
            validation_count=args.validation_count,
            seed=args.seed,
            png_compress_level=args.png_compress_level,
            overwrite=args.overwrite,
            cleanup_archives=args.cleanup_archives_after_build,
        )
    else:
        manifest = build_liu4k_patches(
            raw_root=raw_root,
            output_root=args.output_root,
            patch_size=args.patch_size,
            source_fraction=source_fraction,
            train_count=args.train_count,
            validation_count=args.validation_count,
            seed=args.seed,
            png_compress_level=args.png_compress_level,
            overwrite=args.overwrite,
        )
    payload = {"build": asdict(manifest)}
    if download_manifest is not None:
        payload["download"] = asdict(download_manifest)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
