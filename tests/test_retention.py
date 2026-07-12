from __future__ import annotations

from dataclasses import replace

import pytest

from dqa.web.jobs import JobRecord
from dqa.web.retention import retention_deadlines


def _job(status: str, completed_at: str | None) -> JobRecord:
    return JobRecord(
        job_id="job-1",
        owner_id="user-1",
        status=status,
        dataset_key="uploads/user-1/a.zip",
        preset="detection",
        fail_on="high",
        near_duplicates=False,
        created_at="2026-07-12T00:00:00Z",
        updated_at="2026-07-12T00:00:00Z",
        completed_at=completed_at,
    )


def test_success_retention_deadlines() -> None:
    deadlines = retention_deadlines(_job("succeeded", "2026-07-12T00:00:00Z"))

    assert deadlines.source_expires_at == "2026-07-13T00:00:00Z"
    assert deadlines.artifacts_expire_at == "2026-07-19T00:00:00Z"
    assert deadlines.metadata_expires_at == "2026-08-11T00:00:00Z"


def test_failed_artifacts_expire_after_48_hours() -> None:
    deadlines = retention_deadlines(_job("failed", "2026-07-12T00:00:00Z"))

    assert deadlines.artifacts_expire_at == "2026-07-14T00:00:00Z"


def test_nonterminal_job_has_no_retention_deadlines() -> None:
    with pytest.raises(ValueError, match="terminal"):
        retention_deadlines(_job("running", None))
