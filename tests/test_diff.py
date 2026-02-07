from __future__ import annotations

import json
from pathlib import Path

from dqa.cli import main


def _make_run(path: Path, *, critical: int = 0, high: int = 0, medium: int = 0, low: int = 0, ids: list[tuple[str, str]] | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    findings = []
    if ids:
        for finding_id, sev in ids:
            findings.append({"id": finding_id, "severity": sev, "message": "x", "fingerprint": f"{finding_id}-{sev}-{len(findings)}"})

    summary = {
        "totals": {
            "findings": len(findings),
            "by_severity": {"critical": critical, "high": high, "medium": medium, "low": low},
            "build_failed": False,
        }
    }
    flags = {"findings": findings}
    (path / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (path / "flags.json").write_text(json.dumps(flags), encoding="utf-8")


def test_diff_reports_regressions_and_improvements(tmp_path: Path, capsys) -> None:
    old_run = tmp_path / "old"
    new_run = tmp_path / "new"

    _make_run(
        old_run,
        medium=1,
        low=1,
        ids=[("CLASS_LOW_SUPPORT", "low"), ("BBOX_TINY_BOX", "medium")],
    )
    _make_run(
        new_run,
        medium=2,
        low=0,
        ids=[("BBOX_TINY_BOX", "medium"), ("BBOX_TINY_BOX", "medium")],
    )

    rc = main(["diff", "--old", str(old_run), "--new", str(new_run)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "DQA Diff" in out
    assert "delta=+0" in out
    assert "Top Regressions:" in out
    assert "BBOX_TINY_BOX: +1" in out
    assert "Top Improvements:" in out
    assert "CLASS_LOW_SUPPORT: -1" in out


def test_diff_fail_on_regression_threshold(tmp_path: Path, capsys) -> None:
    old_run = tmp_path / "old"
    new_run = tmp_path / "new"

    _make_run(old_run, medium=0, ids=[])
    _make_run(new_run, medium=1, ids=[("BBOX_TINY_BOX", "medium")])

    rc = main([
        "diff",
        "--old",
        str(old_run),
        "--new",
        str(new_run),
        "--fail-on-regression",
        "medium",
    ])
    err = capsys.readouterr().err

    assert rc == 1
    assert "regression_gate=failed threshold=medium" in err


def test_diff_passes_when_regression_below_threshold(tmp_path: Path, capsys) -> None:
    old_run = tmp_path / "old"
    new_run = tmp_path / "new"

    _make_run(old_run, high=0, low=0, ids=[])
    _make_run(new_run, high=0, low=2, ids=[("CLASS_LOW_SUPPORT", "low"), ("CLASS_LOW_SUPPORT", "low")])

    rc = main([
        "diff",
        "--old",
        str(old_run),
        "--new",
        str(new_run),
        "--fail-on-regression",
        "high",
    ])
    out = capsys.readouterr().out

    assert rc == 0
    assert "regression_gate=passed threshold=high" in out
