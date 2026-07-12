from __future__ import annotations

import base64
import json
from pathlib import Path

from dqa.cli import main


_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)


def _dataset(root: Path, *, with_label: bool) -> Path:
    images = root / "train" / "images"
    labels = root / "train" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (images / "sample.png").write_bytes(_PNG)
    if with_label:
        (labels / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    data = root / "data.yaml"
    data.write_text("path: .\ntrain: train/images\nnames: [object]\n", encoding="utf-8")
    return data


def test_audit_exit_code_matrix(tmp_path: Path, capsys) -> None:
    clean = _dataset(tmp_path / "clean", with_label=True)
    flagged = _dataset(tmp_path / "flagged", with_label=False)

    assert main(["audit", "--data", str(clean), "--out", str(tmp_path / "pass"), "--fail-on", "critical"]) == 0
    assert main(["audit", "--data", str(flagged), "--out", str(tmp_path / "fail"), "--fail-on", "high"]) == 1
    assert main(["audit", "--data", str(clean), "--out", str(tmp_path / "bad"), "--config", str(tmp_path / "missing.yaml")]) == 2
    assert main(["audit", "--data", str(tmp_path / "missing-data.yaml"), "--out", str(tmp_path / "missing")]) == 3

    captured = capsys.readouterr()
    assert "config error:" in captured.err
    assert "runtime error:" in captured.err


def test_repeated_audits_have_deterministic_quality_artifacts(tmp_path: Path) -> None:
    data = _dataset(tmp_path / "dataset", with_label=True)
    first = tmp_path / "first"
    second = tmp_path / "second"

    assert main(["audit", "--data", str(data), "--out", str(first), "--format", "json"]) == 0
    assert main(["audit", "--data", str(data), "--out", str(second), "--format", "json"]) == 0

    assert json.loads((first / "index.json").read_text(encoding="utf-8")) == json.loads(
        (second / "index.json").read_text(encoding="utf-8")
    )
    assert json.loads((first / "flags.json").read_text(encoding="utf-8")) == json.loads(
        (second / "flags.json").read_text(encoding="utf-8")
    )

    first_summary = json.loads((first / "summary.json").read_text(encoding="utf-8"))
    second_summary = json.loads((second / "summary.json").read_text(encoding="utf-8"))
    first_summary.pop("run")
    second_summary.pop("run")
    assert first_summary == second_summary


def test_malformed_coco_returns_config_error_without_traceback(tmp_path: Path, capsys) -> None:
    train = tmp_path / "dataset" / "train"
    train.mkdir(parents=True)
    annotation = train / "_annotations.coco.json"
    annotation.write_text("{not valid json", encoding="utf-8")

    exit_code = main(["audit", "--data", str(annotation), "--out", str(tmp_path / "run")])

    assert exit_code == 2
    assert "config error: Invalid COCO JSON" in capsys.readouterr().err
