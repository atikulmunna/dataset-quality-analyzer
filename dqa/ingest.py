from __future__ import annotations

import os
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


class ArchiveValidationError(ValueError):
    """Raised when an uploaded archive violates hosted ingestion policy."""


@dataclass(frozen=True)
class ArchivePolicy:
    max_archive_bytes: int = 2 * 1024**3
    max_expanded_bytes: int = 10 * 1024**3
    max_entries: int = 100_000
    max_compression_ratio: float = 20.0


@dataclass(frozen=True)
class ValidatedArchive:
    entries: int
    files: int
    compressed_bytes: int
    expanded_bytes: int


def validate_zip(path: Path, policy: ArchivePolicy | None = None) -> ValidatedArchive:
    policy = policy or ArchivePolicy()
    if path.suffix.lower() != ".zip" or not path.is_file():
        raise ArchiveValidationError("Input must be a ZIP file.")
    if path.stat().st_size > policy.max_archive_bytes:
        raise ArchiveValidationError("Archive exceeds compressed size limit.")

    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ArchiveValidationError("Archive is not a valid ZIP file.") from exc

    with archive:
        infos = archive.infolist()
        if len(infos) > policy.max_entries:
            raise ArchiveValidationError("Archive contains too many entries.")

        seen: set[str] = set()
        compressed = 0
        expanded = 0
        files = 0
        for info in infos:
            normalized = _safe_member_name(info)
            if normalized in seen:
                raise ArchiveValidationError("Archive contains duplicate paths.")
            seen.add(normalized)
            if info.flag_bits & 0x1:
                raise ArchiveValidationError("Encrypted ZIP entries are not supported.")
            if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                raise ArchiveValidationError("ZIP compression method is not supported.")
            mode = (info.external_attr >> 16) & 0xFFFF
            if stat.S_ISLNK(mode):
                raise ArchiveValidationError("Archive links are not allowed.")
            if info.is_dir():
                continue
            files += 1
            compressed += info.compress_size
            expanded += info.file_size
            if expanded > policy.max_expanded_bytes:
                raise ArchiveValidationError("Archive exceeds expanded size limit.")
            if expanded > policy.max_compression_ratio * max(1, compressed):
                raise ArchiveValidationError("Archive exceeds compression ratio limit.")

    return ValidatedArchive(
        entries=len(infos),
        files=files,
        compressed_bytes=compressed,
        expanded_bytes=expanded,
    )


def extract_validated_zip(
    path: Path,
    destination: Path,
    policy: ArchivePolicy | None = None,
) -> ValidatedArchive:
    policy = policy or ArchivePolicy()
    report = validate_zip(path, policy)
    if destination.exists():
        raise ArchiveValidationError("Extraction destination must not already exist.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    written = 0
    try:
        with zipfile.ZipFile(path) as archive:
            for info in archive.infolist():
                relative = Path(_safe_member_name(info))
                target = temp_dir / relative
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open("xb") as output:
                    while chunk := source.read(1024 * 1024):
                        written += len(chunk)
                        if written > policy.max_expanded_bytes or written > report.expanded_bytes:
                            raise ArchiveValidationError("Extracted data exceeds validated size.")
                        output.write(chunk)
        if written != report.expanded_bytes:
            raise ArchiveValidationError("Extracted size does not match ZIP metadata.")
        os.replace(temp_dir, destination)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return report


def _safe_member_name(info: zipfile.ZipInfo) -> str:
    name = info.filename
    if not name or "\\" in name or "\x00" in name:
        raise ArchiveValidationError("Archive contains an invalid path.")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ArchiveValidationError("Archive contains an unsafe path.")
    return path.as_posix().rstrip("/")
