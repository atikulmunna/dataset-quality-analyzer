from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from dqa.web.jobs import JobRecord
from dqa.web.lifecycle import JobLifecycle, JobTransitionError, artifact_prefix


class AtomicStore:
    def __init__(self, job: JobRecord) -> None:
        self.job = job

    def get(self, job_id: str) -> JobRecord | None:
        return self.job if self.job.job_id == job_id else None

    def compare_and_swap(self, expected_version: int, replacement: JobRecord) -> bool:
        if self.job.version != expected_version:
            return False
        self.job = replacement
        return True


def _job(*, max_attempts: int = 3) -> JobRecord:
    return JobRecord(
        job_id="job-1",
        owner_id="user-1",
        status="queued",
        dataset_key="uploads/user-1/upload-1/dataset.zip",
        preset="detection",
        fail_on="high",
        near_duplicates=False,
        created_at="2026-07-12T00:00:00Z",
        updated_at="2026-07-12T00:00:00Z",
        max_attempts=max_attempts,
    )


def test_claim_heartbeat_and_idempotent_completion() -> None:
    current = [datetime(2026, 7, 12, tzinfo=timezone.utc)]
    store = AtomicStore(_job())
    lifecycle = JobLifecycle(store, clock=lambda: current[0], lease_seconds=60)

    claimed = lifecycle.claim("job-1", "worker-1")
    assert claimed is not None
    assert claimed.status == "running"
    assert claimed.attempt == 1
    prefix = artifact_prefix(claimed)

    current[0] += timedelta(seconds=30)
    heartbeat = lifecycle.heartbeat("job-1", "worker-1")
    completed = lifecycle.complete("job-1", "worker-1", prefix)
    repeated = lifecycle.complete("job-1", "worker-1", prefix)

    assert heartbeat.lease_until == "2026-07-12T00:01:30Z"
    assert completed.status == "succeeded"
    assert repeated == completed


def test_expired_lease_is_reclaimed_with_new_immutable_attempt_prefix() -> None:
    current = [datetime(2026, 7, 12, tzinfo=timezone.utc)]
    store = AtomicStore(_job())
    lifecycle = JobLifecycle(store, clock=lambda: current[0], lease_seconds=60)
    first = lifecycle.claim("job-1", "worker-1")
    assert first is not None
    first_prefix = artifact_prefix(first)

    current[0] += timedelta(seconds=61)
    second = lifecycle.claim("job-1", "worker-2")
    assert second is not None
    second_prefix = artifact_prefix(second)

    assert second.attempt == 2
    assert first_prefix.endswith("attempt-1/")
    assert second_prefix.endswith("attempt-2/")
    with pytest.raises(JobTransitionError, match="does not own"):
        lifecycle.complete("job-1", "worker-1", first_prefix)
    with pytest.raises(JobTransitionError, match="prefix"):
        lifecycle.complete("job-1", "worker-2", first_prefix)
    assert lifecycle.complete("job-1", "worker-2", second_prefix).status == "succeeded"


def test_failure_requeues_until_attempts_are_exhausted() -> None:
    current = datetime(2026, 7, 12, tzinfo=timezone.utc)
    store = AtomicStore(_job(max_attempts=2))
    lifecycle = JobLifecycle(store, clock=lambda: current)

    assert lifecycle.claim("job-1", "worker-1") is not None
    retry = lifecycle.fail("job-1", "worker-1", "temporary_error")
    assert retry.status == "queued"
    assert retry.completed_at is None

    assert lifecycle.claim("job-1", "worker-2") is not None
    terminal = lifecycle.fail("job-1", "worker-2", "permanent_error")
    assert terminal.status == "failed"
    assert terminal.completed_at is not None
    assert lifecycle.claim("job-1", "worker-3") is None


def test_queued_cancel_is_terminal_and_running_cancel_is_cooperative() -> None:
    current = datetime(2026, 7, 12, tzinfo=timezone.utc)
    queued_store = AtomicStore(_job())
    queued = JobLifecycle(queued_store, clock=lambda: current).request_cancel("user-1", "job-1")
    assert queued is not None and queued.status == "cancelled"

    running_store = AtomicStore(_job())
    lifecycle = JobLifecycle(running_store, clock=lambda: current)
    assert lifecycle.claim("job-1", "worker-1") is not None
    requested = lifecycle.request_cancel("user-1", "job-1")
    assert requested is not None and requested.status == "running" and requested.cancel_requested
    with pytest.raises(JobTransitionError, match="Cancellation"):
        lifecycle.complete("job-1", "worker-1", artifact_prefix(requested))
    cancelled = lifecycle.fail("job-1", "worker-1", "cancelled_by_user")
    assert cancelled.status == "cancelled"


def test_completed_job_cannot_be_overwritten() -> None:
    now = datetime(2026, 7, 12, tzinfo=timezone.utc)
    completed = replace(
        _job(),
        status="succeeded",
        attempt=1,
        worker_id="worker-1",
        result_prefix="artifacts/user-1/job-1/attempt-1/",
        completed_at="2026-07-12T00:00:00Z",
    )
    lifecycle = JobLifecycle(AtomicStore(completed), clock=lambda: now)

    with pytest.raises(JobTransitionError, match="immutable"):
        lifecycle.complete("job-1", "worker-2", "artifacts/user-1/job-1/attempt-2/")


def test_cancel_is_owner_scoped() -> None:
    store = AtomicStore(_job())

    result = JobLifecycle(store).request_cancel("user-2", "job-1")

    assert result is None
    assert store.job.status == "queued"


def test_execution_timeout_spans_retries_and_becomes_terminal() -> None:
    current = [datetime(2026, 7, 12, tzinfo=timezone.utc)]
    store = AtomicStore(_job())
    lifecycle = JobLifecycle(
        store,
        clock=lambda: current[0],
        lease_seconds=60,
        execution_timeout_seconds=120,
    )
    claimed = lifecycle.claim("job-1", "worker-1")
    assert claimed is not None
    assert claimed.execution_started_at == "2026-07-12T00:00:00Z"

    current[0] += timedelta(seconds=121)
    timed_out = lifecycle.expire_timed_out("job-1")

    assert timed_out is not None
    assert timed_out.status == "failed"
    assert timed_out.error_code == "execution_timeout"
    assert lifecycle.claim("job-1", "worker-2") is None
