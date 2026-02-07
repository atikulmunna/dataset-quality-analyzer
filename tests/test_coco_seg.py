from __future__ import annotations

import json
from pathlib import Path

from dqa.indexer import build_index_from_coco


def test_build_index_from_coco_segmentation(tmp_path: Path) -> None:
    train = tmp_path / "train"
    train.mkdir(parents=True, exist_ok=True)

    img = train / "img1.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")

    coco = {
        "images": [{"id": 1, "file_name": "img1.jpg", "width": 100, "height": 100}],
        "categories": [{"id": 5, "name": "road"}],
        "annotations": [
            {
                "id": 11,
                "image_id": 1,
                "category_id": 5,
                "segmentation": [[10, 10, 90, 10, 90, 90, 10, 90]],
                "bbox": [10, 10, 80, 80],
            }
        ],
    }
    (train / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")

    result = build_index_from_coco(tmp_path, requested_splits=["train"])
    assert result.class_count == 1
    row = result.payload["images"][0]
    assert row["split"] == "train"
    assert row["label_rows"][0]["annotation_type"] == "segment"
    assert row["label_rows"][0]["class_id"] == 0
    assert max(row["label_rows"][0]["coords"]) <= 1.0


def test_build_index_from_coco_bbox_fallback(tmp_path: Path) -> None:
    valid = tmp_path / "valid"
    valid.mkdir(parents=True, exist_ok=True)

    img = valid / "img2.jpg"
    img.write_bytes(b"\xff\xd8\xff\xd9")

    coco = {
        "images": [{"id": 2, "file_name": "img2.jpg", "width": 200, "height": 100}],
        "categories": [{"id": 1, "name": "car"}],
        "annotations": [{"id": 21, "image_id": 2, "category_id": 1, "bbox": [20, 10, 40, 20]}],
    }
    (valid / "_annotations.coco.json").write_text(json.dumps(coco), encoding="utf-8")

    result = build_index_from_coco(tmp_path, requested_splits=["val"])
    row = result.payload["images"][0]
    ann = row["label_rows"][0]
    assert row["split"] == "val"
    assert ann["annotation_type"] == "bbox"
    assert 0 <= ann["x_center"] <= 1
    assert 0 <= ann["width"] <= 1
