import io
import os
import re
import threading
import zipfile
from bisect import bisect_right
from contextlib import contextmanager
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
SPLIT_PART_PATTERN = re.compile(r"\.z\d+", re.IGNORECASE)
_ZIPFILE_PATCH_LOCK = threading.Lock()
DEFAULT_LIU4K_GDRIVE_URLS = {
    "train": "https://drive.google.com/drive/folders/1FtVQtY2t_ecuy_gzJqZ-CatqrJBAdq_d",
    "validation": "https://drive.google.com/drive/folders/1OCSXbWAlZ_im9oVIocOlr8tTjKFlOYM-",
}


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
        if SPLIT_PART_PATTERN.fullmatch(candidate.suffix):
            parts.append(candidate)
    return sorted(parts, key=_split_part_index)


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
                fpin.seek(offset - zipfile.sizeEndCentDir64Locator, 2)
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
                offset - zipfile.sizeEndCentDir64Locator - zipfile.sizeEndCentDir64, 2
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
                dircount,
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


def _archive_parts(archive_path: Path) -> list[Path]:
    split_parts = _find_split_parts(archive_path)
    suffix = archive_path.suffix.lower()

    if suffix == ".zip":
        return [*split_parts, archive_path] if split_parts else [archive_path]

    if SPLIT_PART_PATTERN.fullmatch(archive_path.suffix):
        if split_parts:
            return split_parts
        return [archive_path]

    return [archive_path]


def _part_offsets(parts: list[Path]) -> list[int]:
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


def _patch_split_member_offsets(zf: ZipFile, parts: list[Path]) -> None:
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

        candidates = [
            current_offset,
            current_offset + volume_shift,
            current_offset - first_raw_offset + split_prefix_shift,
            current_offset - first_raw_offset + volume_shift + split_prefix_shift,
        ]

        seen = set()
        for candidate_offset in candidates:
            if candidate_offset in seen:
                continue
            seen.add(candidate_offset)

            if _has_local_file_header(zf, candidate_offset):
                info.header_offset = candidate_offset
                break


class _SplitZipStream(io.BufferedIOBase):

    def __init__(self, parts: list[Path]) -> None:
        super().__init__()
        if not parts:
            raise ValueError("Split archive requires at least one part")

        self._files = [part.open("rb") for part in parts]
        self._sizes = [part.stat().st_size for part in parts]
        self._offsets = []
        current = 0
        for size in self._sizes:
            self._offsets.append(current)
            current += size
        self._total_size = current
        self._position = 0

    def _locate_part(self, absolute_offset: int) -> int:
        if absolute_offset >= self._total_size:
            return len(self._offsets) - 1
        idx = bisect_right(self._offsets, absolute_offset) - 1
        return max(idx, 0)

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
        chunks = []

        while remaining > 0:
            part_idx = self._locate_part(self._position)
            part_start = self._offsets[part_idx]
            offset_inside_part = self._position - part_start
            available_in_part = self._sizes[part_idx] - offset_inside_part
            to_read = min(remaining, available_in_part)

            f = self._files[part_idx]
            f.seek(offset_inside_part)
            chunk = f.read(to_read)
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
        for f in self._files:
            f.close()
        super().close()


@contextmanager
def _open_archive_zip(zip_path: Path):
    archive_parts = _archive_parts(zip_path)
    if len(archive_parts) == 1 and archive_parts[0] == zip_path:
        with ZipFile(zip_path, "r") as zf:
            yield zf
        return

    stream = _SplitZipStream(archive_parts)
    try:
        with _allow_multidisk_zipfile():
            with ZipFile(stream, "r") as zf:
                _patch_split_member_offsets(zf, archive_parts)
                yield zf
    finally:
        stream.close()


def _iter_archives(all_files: list[Path]):
    by_stem: dict[Path, list[Path]] = {}
    for path in all_files:
        if path.suffix.lower() == ".zip" or SPLIT_PART_PATTERN.fullmatch(path.suffix):
            stem = path.with_suffix("")
            by_stem.setdefault(stem, []).append(path)

    for stem, parts in by_stem.items():
        zip_candidate = next((p for p in parts if p.suffix.lower() == ".zip"), None)
        if zip_candidate is not None:
            yield zip_candidate
            continue

        split_only_parts = sorted(
            [p for p in parts if SPLIT_PART_PATTERN.fullmatch(p.suffix)],
            key=_split_part_index,
        )
        if split_only_parts:
            yield split_only_parts[-1]


def _archive_class_name(path: Path) -> str:
    if path.suffix.lower() == ".zip":
        return path.stem
    if SPLIT_PART_PATTERN.fullmatch(path.suffix):
        return path.with_suffix("").name
    return path.stem


def _iter_zip_image_chains(
    zf: ZipFile,
    prefix: tuple[str, ...] = (),
    depth: int = 0,
    max_nested_zip_depth: int = 2,
):
    for info in zf.infolist():
        if info.is_dir():
            continue

        member_name = info.filename.replace("\\", "/")
        chain = prefix + (member_name,)

        if _is_image_file(member_name):
            yield chain
            continue

        if not member_name.lower().endswith(".zip"):
            continue

        if depth >= max_nested_zip_depth:
            continue

        try:
            nested_bytes = zf.read(info)
            with ZipFile(io.BytesIO(nested_bytes), "r") as nested_zip:
                yield from _iter_zip_image_chains(
                    nested_zip,
                    prefix=chain,
                    depth=depth + 1,
                    max_nested_zip_depth=max_nested_zip_depth,
                )
        except BadZipFile:
            continue


def _to_grayscale_tensor(image: Image.Image) -> torch.Tensor:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    gray = rgb.mean(axis=2)
    return torch.from_numpy(gray)


def _looks_like_folder_url(url: str) -> bool:
    lowered = url.lower()
    return "drive.google.com" in lowered and "folders" in lowered


def _download_liu4k_from_gdrive(
    destination_dir: Path,
    gdrive_url: str | None,
    gdrive_file_id: str | None,
    output_name: str,
) -> None:
    try:
        import gdown
    except ImportError as exc:
        raise RuntimeError(
            "Auto-download requires `gdown`. Install it with: pip install gdown"
        ) from exc

    destination_dir.mkdir(parents=True, exist_ok=True)

    if gdrive_url:
        if _looks_like_folder_url(gdrive_url):
            downloaded = gdown.download_folder(
                url=gdrive_url,
                output=str(destination_dir),
                quiet=False,
            )
            if not downloaded:
                raise RuntimeError("Google Drive folder download returned no files")
            return

        archive_path = destination_dir / output_name
        downloaded_path = gdown.download(
            url=gdrive_url,
            output=str(archive_path),
            quiet=False,
            fuzzy=True,
        )
    elif gdrive_file_id:
        archive_path = destination_dir / output_name
        downloaded_path = gdown.download(
            id=gdrive_file_id,
            output=str(archive_path),
            quiet=False,
        )
    else:
        raise RuntimeError("Neither gdrive_url nor gdrive_file_id was provided")

    if downloaded_path is None:
        raise RuntimeError("Google Drive download failed")

    downloaded_file = Path(downloaded_path)
    if downloaded_file.suffix.lower() == ".zip":
        with ZipFile(downloaded_file, "r") as zf:
            zf.extractall(destination_dir)


def _directory_has_files(directory: Path) -> bool:
    try:
        next(directory.iterdir())
        return True
    except StopIteration:
        return False


def _normalize_split_name(split: str | None) -> str | None:
    if split is None:
        return None
    normalized = split.strip().lower()
    aliases = {
        "train": "train",
        "training": "train",
        "tr": "train",
        "val": "validation",
        "valid": "validation",
        "validation": "validation",
        "dev": "validation",
    }
    return aliases.get(normalized, normalized)


def _guess_split_from_root(root: Path) -> str | None:
    candidates = [root.name.lower(), root.parent.name.lower()]
    for value in candidates:
        guessed = _normalize_split_name(value)
        if guessed in DEFAULT_LIU4K_GDRIVE_URLS:
            return guessed
    return None


def _resolve_gdrive_source(
    root: Path,
    split: str | None,
    gdrive_url: str | None,
    gdrive_file_id: str | None,
) -> tuple[str | None, str | None]:
    resolved_url = gdrive_url or os.getenv("LIU4K_GDRIVE_URL")
    resolved_file_id = gdrive_file_id or os.getenv("LIU4K_GDRIVE_FILE_ID")

    if resolved_url or resolved_file_id:
        return resolved_url, resolved_file_id

    resolved_split = _normalize_split_name(split) or _guess_split_from_root(root)
    if resolved_split in DEFAULT_LIU4K_GDRIVE_URLS:
        return DEFAULT_LIU4K_GDRIVE_URLS[resolved_split], None

    return None, None


class LIU4KDataset(Dataset):

    def __init__(
        self,
        root: str | Path,
        recursive: bool = True,
        return_class: bool = False,
        cache_dir: str | Path | None = None,
        max_nested_zip_depth: int = 0,
        auto_download_if_empty: bool = True,
        split: str | None = None,
        gdrive_url: str | None = None,
        gdrive_file_id: str | None = None,
        download_output_name: str = "liu4k_download.zip",
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.auto_download_if_empty = auto_download_if_empty
        self.split = _normalize_split_name(split)
        self.gdrive_url, self.gdrive_file_id = _resolve_gdrive_source(
            root=self.root,
            split=self.split,
            gdrive_url=gdrive_url,
            gdrive_file_id=gdrive_file_id,
        )
        self.download_output_name = download_output_name

        self._zip_handles: dict[Path, ZipFile] = {}
        self._zip_streams: dict[Path, _SplitZipStream] = {}

        self._ensure_dataset_present()

        self.recursive = recursive
        self.return_class = return_class
        self.max_nested_zip_depth = max_nested_zip_depth
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None

        self.samples: list[tuple[str, str, int]] = []

        self.class_to_idx: dict[str, int] = {}
        self.classes: list[str] = []

        self._index_dataset()

        if not self.samples:
            raise RuntimeError(f"No images found in {self.root}")

    def _ensure_dataset_present(self) -> None:
        if not self.root.exists():
            if not self.auto_download_if_empty:
                raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
            self.root.mkdir(parents=True, exist_ok=True)

        if _directory_has_files(self.root):
            return

        if not self.auto_download_if_empty:
            return

        if not self.gdrive_url and not self.gdrive_file_id:
            raise RuntimeError(
                "Dataset folder is empty and auto-download is enabled, but no Google Drive "
                "source was provided. Pass gdrive_url/gdrive_file_id or set "
                "LIU4K_GDRIVE_URL / LIU4K_GDRIVE_FILE_ID. You can also pass split='train' "
                "or split='validation' for built-in LIU4K links."
            )

        _download_liu4k_from_gdrive(
            destination_dir=self.root,
            gdrive_url=self.gdrive_url,
            gdrive_file_id=self.gdrive_file_id,
            output_name=self.download_output_name,
        )

        if not _directory_has_files(self.root):
            raise RuntimeError(
                f"Auto-download completed but dataset folder is still empty: {self.root}"
            )

    def _class_index(self, class_name: str) -> int:
        if class_name not in self.class_to_idx:
            self.class_to_idx[class_name] = len(self.classes)
            self.classes.append(class_name)
        return self.class_to_idx[class_name]

    def _iter_paths(self) -> list[Path]:
        if self.recursive:
            files = [p for p in self.root.rglob("*") if p.is_file()]
        else:
            files = [p for p in self.root.iterdir() if p.is_file()]
        return files

    def _index_dataset(self) -> None:
        all_files = self._iter_paths()

        for p in all_files:
            if _is_image_file(p.name):
                class_name = p.parent.name
                class_idx = self._class_index(class_name)
                self.samples.append(("file", str(p.resolve()), class_idx))

        archive_files = list(_iter_archives(all_files))
        for archive_path in archive_files:
            class_name = _archive_class_name(archive_path)
            class_idx = self._class_index(class_name)
            try:
                with _open_archive_zip(archive_path) as zf:
                    for chain in _iter_zip_image_chains(
                        zf,
                        max_nested_zip_depth=self.max_nested_zip_depth,
                    ):
                        ref = f"{archive_path.resolve()}::{'||'.join(chain)}"
                        self.samples.append(("zip", ref, class_idx))
            except BadZipFile as exc:
                raise RuntimeError(f"Broken zip archive: {archive_path}") from exc

    def __len__(self) -> int:
        return len(self.samples)

    def _get_zip(self, path: Path) -> ZipFile:
        if path not in self._zip_handles:
            archive_parts = _archive_parts(path)
            if len(archive_parts) == 1 and archive_parts[0] == path:
                self._zip_handles[path] = ZipFile(path, "r")
            else:
                stream = _SplitZipStream(archive_parts)
                self._zip_streams[path] = stream
                with _allow_multidisk_zipfile():
                    self._zip_handles[path] = ZipFile(stream, "r")
                _patch_split_member_offsets(self._zip_handles[path], archive_parts)
        return self._zip_handles[path]

    def __getitem__(self, index: int) -> torch.Tensor | tuple[torch.Tensor, int]:
        source_type, source_ref, class_idx = self.samples[index]

        if source_type == "file":
            with Image.open(source_ref) as img:
                x = _to_grayscale_tensor(img)
        else:
            zip_path_str, chain_str = source_ref.split("::", 1)
            zip_path = Path(zip_path_str)
            zf = self._get_zip(zip_path)

            member_chain = chain_str.split("||")
            current_zip = zf
            nested_zip_handles: list[ZipFile] = []
            raw = None

            try:
                for i, member_name in enumerate(member_chain):
                    is_last = i == len(member_chain) - 1
                    with current_zip.open(member_name, "r") as f:
                        data = f.read()

                    if is_last:
                        raw = data
                        break

                    nested_zip = ZipFile(io.BytesIO(data), "r")
                    nested_zip_handles.append(nested_zip)
                    current_zip = nested_zip
            finally:
                for nested_zip in nested_zip_handles:
                    nested_zip.close()

            if raw is None:
                raise RuntimeError(
                    f"Failed to load sample at index {index}: {source_ref}"
                )

            with Image.open(io.BytesIO(raw)) as img:
                x = _to_grayscale_tensor(img)

        if self.return_class:
            return x, class_idx
        return x

    def close(self) -> None:
        zip_handles = getattr(self, "_zip_handles", {})
        for zf in zip_handles.values():
            zf.close()
        zip_handles.clear()

        zip_streams = getattr(self, "_zip_streams", {})
        for stream in zip_streams.values():
            stream.close()
        zip_streams.clear()

    def __del__(self) -> None:
        self.close()
