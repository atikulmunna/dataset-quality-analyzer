from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from dqa.cli import main


_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)
_ROOT = Path(__file__).resolve().parents[1]


def _assert_artifacts(out: Path) -> None:
    flags = json.loads((out / "flags.json").read_text(encoding="utf-8"))
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))

    assert flags["schema_version"] == "1.0.0"
    assert summary["schema_version"] == "1.0.0"
    assert summary["dataset"]["splits"]["train"]["images"] == 1

    flags_schema = json.loads((_ROOT / "schemas" / "flags.schema.json").read_text(encoding="utf-8"))
    summary_schema = json.loads((_ROOT / "schemas" / "summary.schema.json").read_text(encoding="utf-8"))
    Draft202012Validator(flags_schema).validate(flags)
    Draft202012Validator(summary_schema).validate(summary)


@pytest.mark.parametrize(
    "label",
    [
        "0 0.5 0.5 0.5 0.5\n",
        "0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9\n",
    ],
    ids=["detection", "segmentation"],
)
def test_yolo_formats_audit_end_to_end(tmp_path: Path, label: str) -> None:
    dataset = tmp_path / "dataset"
    images = dataset / "train" / "images"
    labels = dataset / "train" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (images / "sample.png").write_bytes(_PNG)
    (labels / "sample.txt").write_text(label, encoding="utf-8")
    data_yaml = dataset / "data.yaml"
    data_yaml.write_text("path: .\ntrain: train/images\nnames: [object]\n", encoding="utf-8")
    out = tmp_path / "run"

    exit_code = main(
        [
            "audit",
            "--data",
            str(data_yaml),
            "--out",
            str(out),
            "--config",
            str(_ROOT / "dqa.yaml"),
            "--format",
            "json",
            "--fail-on",
            "critical",
        ]
    )

    assert exit_code == 0
    _assert_artifacts(out)


@pytest.mark.parametrize("source_kind", ["directory", "json"], ids=["directory", "single-json"])
def test_coco_inputs_audit_end_to_end(tmp_path: Path, source_kind: str) -> None:
    dataset = tmp_path / "dataset"
    train = dataset / "train"
    train.mkdir(parents=True)
    (train / "sample.png").write_bytes(_PNG)
    annotations = train / "_annotations.coco.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "sample.png", "width": 1, "height": 1}],
                "categories": [{"id": 1, "name": "object"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "segmentation": [[0.1, 0.1, 0.9, 0.1, 0.9, 0.9]],
                        "bbox": [0.1, 0.1, 0.8, 0.8],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    source = dataset if source_kind == "directory" else annotations
    out = tmp_path / "run"

    exit_code = main(
        [
            "audit",
            "--data",
            str(source),
            "--out",
            str(out),
            "--config",
            str(_ROOT / "dqa_seg.yaml"),
            "--format",
            "json",
            "--fail-on",
            "critical",
        ]
    )

    assert exit_code == 0
    _assert_artifacts(out)
