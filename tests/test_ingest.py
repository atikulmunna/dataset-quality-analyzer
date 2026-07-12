from __future__ import annotations

import stat
import zipfile
from pathlib import Path

import pytest

from dqa.ingest import ArchivePolicy, ArchiveValidationError, _safe_member_name, extract_validated_zip, validate_zip


def _zip(path: Path, entries: dict[str, bytes], *, compression: int = zipfile.ZIP_STORED) -> Path:
    with zipfile.ZipFile(path, "w", compression=compression) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return path


def test_safe_zip_validates_and_extracts_atomically(tmp_path: Path) -> None:
    archive = _zip(
        tmp_path / "dataset.zip",
        {"data.yaml": b"path: .\n", "train/labels/a.txt": b"0 0.5 0.5 0.2 0.2\n"},
    )
    destination = tmp_path / "dataset"

    report = extract_validated_zip(archive, destination)

    assert report.files == 2
    assert (destination / "data.yaml").read_text(encoding="utf-8") == "path: .\n"
    assert not list(tmp_path.glob(".dataset.*"))


@pytest.mark.parametrize("name", ["../escape.txt", "/absolute.txt"])
def test_archive_paths_fail_closed(tmp_path: Path, name: str) -> None:
    archive = _zip(tmp_path / "dataset.zip", {name: b"bad"})

    with pytest.raises(ArchiveValidationError, match="path"):
        validate_zip(archive)


def test_backslash_member_path_fails_closed() -> None:
    info = zipfile.ZipInfo("safe.txt")
    info.filename = "train\\evil.txt"

    with pytest.raises(ArchiveValidationError, match="path"):
        _safe_member_name(info)


def test_archive_rejects_links(tmp_path: Path) -> None:
    archive_path = tmp_path / "dataset.zip"
    link = zipfile.ZipInfo("train/link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(link, "../../outside")

    with pytest.raises(ArchiveValidationError, match="links"):
        validate_zip(archive_path)


def test_archive_rejects_compression_bomb_ratio(tmp_path: Path) -> None:
    archive = _zip(
        tmp_path / "dataset.zip",
        {"large.txt": b"0" * 100_000},
        compression=zipfile.ZIP_DEFLATED,
    )

    with pytest.raises(ArchiveValidationError, match="compression ratio"):
        validate_zip(archive, ArchivePolicy(max_compression_ratio=5))


def test_archive_rejects_entry_and_size_limits(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "dataset.zip", {"a.txt": b"123", "b.txt": b"456"})

    with pytest.raises(ArchiveValidationError, match="too many entries"):
        validate_zip(archive, ArchivePolicy(max_entries=1))
    with pytest.raises(ArchiveValidationError, match="expanded size"):
        validate_zip(archive, ArchivePolicy(max_expanded_bytes=5, max_compression_ratio=100))


def test_archive_rejects_encrypted_flag(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "dataset.zip", {"a.txt": b"data"})
    payload = bytearray(archive.read_bytes())
    local = payload.find(b"PK\x03\x04")
    central = payload.find(b"PK\x01\x02")
    payload[local + 6] |= 0x1
    payload[central + 8] |= 0x1
    archive.write_bytes(payload)

    with pytest.raises(ArchiveValidationError, match="Encrypted"):
        validate_zip(archive)


def test_extraction_refuses_existing_destination(tmp_path: Path) -> None:
    archive = _zip(tmp_path / "dataset.zip", {"a.txt": b"data"})
    destination = tmp_path / "dataset"
    destination.mkdir()

    with pytest.raises(ArchiveValidationError, match="must not already exist"):
        extract_validated_zip(archive, destination)
