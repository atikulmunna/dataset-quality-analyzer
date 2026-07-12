from __future__ import annotations

import base64
import json
from pathlib import Path

from dqa.audit import AuditOptions, audit_dataset
from dqa.cli import main


_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR4nGNgYAAAAAMAASsJTYQAAAAASUVORK5CYII="
)


def _dataset(root: Path) -> Path:
    images = root / "train" / "images"
    labels = root / "train" / "labels"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (images / "sample.png").write_bytes(_PNG)
    (labels / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    data = root / "data.yaml"
    data.write_text("path: .\ntrain: train/images\nnames: [object]\n", encoding="utf-8")
    return data


def test_service_and_cli_produce_equivalent_quality_artifacts(tmp_path: Path) -> None:
    data = _dataset(tmp_path / "dataset")
    service_out = tmp_path / "service"
    cli_out = tmp_path / "cli"

    service_result = audit_dataset(
        AuditOptions(data=data, out=service_out, formats=("json",), workers=1)
    )
    cli_exit = main(
        [
            "audit",
            "--data",
            str(data),
            "--out",
            str(cli_out),
            "--format",
            "json",
            "--workers",
            "1",
        ]
    )

    assert cli_exit == service_result.exit_code
    assert json.loads((cli_out / "index.json").read_text(encoding="utf-8")) == service_result.index
    assert json.loads((cli_out / "flags.json").read_text(encoding="utf-8")) == service_result.flags

    cli_summary = json.loads((cli_out / "summary.json").read_text(encoding="utf-8"))
    service_summary = dict(service_result.summary)
    cli_summary.pop("run")
    service_summary.pop("run")
    assert cli_summary == service_summary


def test_service_returns_artifacts_without_parsing_cli_output(tmp_path: Path) -> None:
    data = _dataset(tmp_path / "dataset")

    result = audit_dataset(AuditOptions(data=data, out=tmp_path / "run", formats=("json",)))

    assert result.exit_code == 0
    assert result.summary["schema_version"] == "1.0.0"
    assert result.flags["schema_version"] == "1.0.0"
    assert result.index["schema_version"] == "1.0.0"
