from pathlib import Path

from dqa.checks.integrity import run_integrity
from dqa.indexer import _parse_label_rows


def test_parse_yolo_segment_row(tmp_path: Path) -> None:
    label = tmp_path / "seg.txt"
    label.write_text("0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n", encoding="utf-8")

    rows, errors = _parse_label_rows(label)

    assert not errors
    assert len(rows) == 1
    assert rows[0]["annotation_type"] == "segment"
    assert rows[0]["class_id"] == 0
    assert 0 <= rows[0]["width"] <= 1
    assert 0 <= rows[0]["height"] <= 1


def test_integrity_segment_out_of_range_detected() -> None:
    index_payload = {
        "splits": {"train": {"orphan_labels": []}},
        "images": [
            {
                "split": "train",
                "image": "train/images/a.jpg",
                "label": "train/labels/a.txt",
                "label_exists": True,
                "label_rows": [
                    {
                        "line": 1,
                        "annotation_type": "segment",
                        "class_id": 0,
                        "coords": [0.1, 0.1, 1.2, 0.2, 0.2, 0.2],
                        "x_center": 0.0,
                        "y_center": 0.0,
                        "width": 0.0,
                        "height": 0.0,
                    }
                ],
                "label_parse_errors": [],
                "image_error": None,
            }
        ],
    }

    findings = run_integrity(index_payload, class_count=1)
    ids = [f.id for f in findings]
    assert "INTEGRITY_COORD_OUT_OF_RANGE" in ids
