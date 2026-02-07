from __future__ import annotations

import json
from pathlib import Path

from dqa.cli import main


def test_validate_summary_schema_success(tmp_path: Path, capsys) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "schema_version": "1.0.0",
        "run": {
            "run_id": "x",
            "dqa_version": "0.1.0",
            "started_at": "2026-02-06T00:00:00Z",
            "finished_at": "2026-02-06T00:00:01Z",
            "duration_sec": 1.0,
            "config": {"fail_on": "medium", "enabled_checks": ["integrity"]},
        },
        "dataset": {
            "data_yaml": "data.yaml",
            "root": ".",
            "splits": {"train": {"images": 1, "labeled": 1, "unlabeled": 0}},
            "classes": {"count": 1, "names": ["a"]},
        },
        "checks": {
            "integrity": {
                "status": "completed",
                "counts": {"critical": 0, "high": 0, "medium": 0, "low": 0},
            }
        },
        "totals": {
            "findings": 0,
            "by_severity": {"critical": 0, "high": 0, "medium": 0, "low": 0},
            "fail_threshold": "medium",
            "build_failed": False,
        },
    }
    artifact = run_dir / "summary.json"
    artifact.write_text(json.dumps(summary), encoding="utf-8")

    rc = main(["validate", "--artifact", str(artifact), "--schema", "schemas/summary.schema.json"])
    out = capsys.readouterr().out

    assert rc == 0
    assert "valid artifact=" in out


def test_validate_summary_schema_failure(tmp_path: Path, capsys) -> None:
    artifact = tmp_path / "summary.json"
    artifact.write_text(json.dumps({"schema_version": "1.0.0"}), encoding="utf-8")

    rc = main(["validate", "--artifact", str(artifact), "--schema", "schemas/summary.schema.json"])
    err = capsys.readouterr().err

    assert rc == 1
    assert "invalid artifact=" in err


def test_validate_invalid_schema_returns_config_error(tmp_path: Path, capsys) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text(json.dumps({"a": 1}), encoding="utf-8")

    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"type": 3}), encoding="utf-8")

    rc = main(["validate", "--artifact", str(artifact), "--schema", str(schema)])
    err = capsys.readouterr().err

    assert rc == 2
    assert "config error: Invalid JSON schema" in err
