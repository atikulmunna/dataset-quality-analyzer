from __future__ import annotations

import pytest

from web_dashboard import DashboardError, _build_audit, _build_diff, _build_explain, _build_validate


def test_dashboard_audit_keeps_user_values_as_single_arguments() -> None:
    suspicious_path = 'dataset; echo "not executed"'
    cmd, env = _build_audit(
        {
            "audit_data": [suspicious_path],
            "audit_out": ["runs/test output"],
            "audit_config": ["dqa.yaml"],
            "audit_roboflow_api_key": ["secret"],
        }
    )

    assert cmd[cmd.index("--data") + 1] == suspicious_path
    assert cmd[cmd.index("--out") + 1] == "runs/test output"
    assert "secret" not in cmd
    assert env == {"ROBOFLOW_API_KEY": "secret"}


def test_dashboard_audit_rejects_missing_or_ambiguous_source() -> None:
    with pytest.raises(DashboardError):
        _build_audit({})
    with pytest.raises(DashboardError):
        _build_audit({"audit_data": ["data.yaml"], "audit_data_url": ["https://example.com"]})


def test_dashboard_secondary_command_builders() -> None:
    explain, _ = _build_explain({"explain_run": ["runs/a"]})
    validate, _ = _build_validate({"validate_artifact": ["a.json"], "validate_schema": ["a.schema.json"]})
    diff, _ = _build_diff({"diff_old": ["runs/a"], "diff_new": ["runs/b"]})

    assert explain[-2:] == ["--format", "text"]
    assert validate[-4:] == ["--artifact", "a.json", "--schema", "a.schema.json"]
    assert diff[-4:] == ["--old", "runs/a", "--new", "runs/b"]
