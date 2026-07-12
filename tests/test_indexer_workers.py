from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from dqa.indexer import build_index
from dqa.io_yolo import load_dataset_spec


def test_yolo_hashing_uses_bounded_parallel_workers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    images = tmp_path / "train" / "images"
    labels = tmp_path / "train" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for index in range(4):
        (images / f"sample_{index}.png").write_bytes(b"image")
        (labels / f"sample_{index}.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    data = tmp_path / "data.yaml"
    data.write_text("path: .\ntrain: train/images\nnames: [object]\n", encoding="utf-8")

    lock = threading.Lock()
    active = 0
    maximum_active = 0

    def observed_hash(_path: Path) -> str:
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return "hash"

    monkeypatch.setattr("dqa.indexer._sha256_file", observed_hash)

    result = build_index(load_dataset_spec(data), workers=2)

    assert result.cache_misses == 4
    assert maximum_active == 2
