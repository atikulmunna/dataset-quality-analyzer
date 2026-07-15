from __future__ import annotations

from pathlib import Path

import pytest

from dqa.aws.worker import discover_dataset


def test_discover_dataset_prefers_single_yolo_entry(tmp_path: Path) -> None:
    data = tmp_path / "export" / "data.yaml"
    data.parent.mkdir()
    data.write_text("names: []\n", encoding="utf-8")
    (tmp_path / "export" / "metadata.json").write_text("{}", encoding="utf-8")

    assert discover_dataset(tmp_path) == data


def test_discover_dataset_rejects_ambiguous_yolo_entries(tmp_path: Path) -> None:
    for name in ("one", "two"):
        path = tmp_path / name / "data.yaml"
        path.parent.mkdir()
        path.write_text("names: []\n", encoding="utf-8")

    with pytest.raises(ValueError, match="multiple"):
        discover_dataset(tmp_path)


def test_discover_dataset_accepts_coco_directory(tmp_path: Path) -> None:
    annotation = tmp_path / "train" / "_annotations.coco.json"
    annotation.parent.mkdir()
    annotation.write_text("{}", encoding="utf-8")

    assert discover_dataset(tmp_path) == tmp_path


def test_discover_dataset_requires_supported_entry(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="neither"):
        discover_dataset(tmp_path)

