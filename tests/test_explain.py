from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dqa.cli import ExplainError, run_explain


def test_explain_low_only_summary(capsys: pytest.CaptureFixture[str], tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "totals": {
            "build_failed": False,
            "by_severity": {"critical": 0, "high": 0, "medium": 0, "low": 3},
        }
    }
    flags = {
        "findings": [
            {"id": "CLASS_LOW_SUPPORT", "severity": "low", "message": "x"},
            {"id": "CLASS_LOW_SUPPORT", "severity": "low", "message": "x"},
            {"id": "CLASS_LOW_SUPPORT", "severity": "low", "message": "x"},
        ]
    }
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "flags.json").write_text(json.dumps(flags), encoding="utf-8")

    rc = run_explain(SimpleNamespace(run=run_dir, summary=None, flags=None))
    out = capsys.readouterr().out

    assert rc == 0
    assert "DQA Explain" in out
    assert "CLASS_LOW_SUPPORT" in out
    assert "Low-severity only" in out


def test_explain_requires_inputs(tmp_path: Path) -> None:
    with pytest.raises(ExplainError):
        run_explain(SimpleNamespace(run=None, summary=None, flags=None))
