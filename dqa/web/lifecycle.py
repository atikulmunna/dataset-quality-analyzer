from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Callable, Protocol

from .jobs import JobRecord


class JobTransitionError(RuntimeError):
    """Raised when a requested job transition is invalid or stale."""


class LifecycleStore(Protocol):
    def get(self, job_id: str) -> JobRecord | None: ...

    def compare_and_swap(self, expected_version: int, replacement: JobRecord) -> bool: ...


class JobLifecycle:
    def __init__(
        self,
        store: LifecycleStore,
        *,
        clock: Callable[[], datetime] | None = None,
        lease_seconds: int = 300,
        execution_timeout_seconds: int = 7200,
    ) -> None:
        self._store = store
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lease_seconds = lease_seconds
        self._execution_timeout_seconds = execution_timeout_seconds

    def claim(self, job_id: str, worker_id: str) -> JobRecord | None:
        now = self._now()
        for _ in range(3):
            job = self._required(job_id)
            if self._is_timed_out(job, now):
                timed_out = self._transition(
                    job,
                    status="failed",
                    error_code="execution_timeout",
                    completed_at=_format_time(now),
                    lease_until=None,
                )
                if self._store.compare_and_swap(job.version, timed_out):
                    return None
                continue
            reclaiming = job.status == "running" and _parse_time(job.lease_until) <= now
            if job.status != "queued" and not reclaiming:
                return None
            if job.cancel_requested:
                cancelled = self._transition(job, status="cancelled", completed_at=_format_time(now))
                if self._store.compare_and_swap(job.version, cancelled):
                    return None
                continue
            if job.attempt >= job.max_attempts:
                failed = self._transition(
                    job,
                    status="failed",
                    error_code="attempts_exhausted",
                    completed_at=_format_time(now),
                )
                if self._store.compare_and_swap(job.version, failed):
                    return None
                continue
            claimed = self._transition(
                job,
                status="running",
                attempt=job.attempt + 1,
                worker_id=worker_id,
                lease_until=_format_time(now + timedelta(seconds=self._lease_seconds)),
                error_code=None,
                execution_started_at=job.execution_started_at or _format_time(now),
            )
            if self._store.compare_and_swap(job.version, claimed):
                return claimed
        raise JobTransitionError("Job changed concurrently while claiming.")

    def heartbeat(self, job_id: str, worker_id: str) -> JobRecord:
        now = self._now()
        job = self._required(job_id)
        self._require_active_worker(job, worker_id, now)
        updated = self._transition(
            job,
            lease_until=_format_time(now + timedelta(seconds=self._lease_seconds)),
        )
        return self._swap(job, updated, "heartbeat")

    def complete(self, job_id: str, worker_id: str, result_prefix: str) -> JobRecord:
        job = self._required(job_id)
        if job.status == "succeeded":
            if job.worker_id == worker_id and job.result_prefix == result_prefix:
                return job
            raise JobTransitionError("Completed job is immutable.")
        now = self._now()
        self._require_active_worker(job, worker_id, now)
        if job.cancel_requested:
            raise JobTransitionError("Cancellation was requested.")
        expected_prefix = artifact_prefix(job)
        if result_prefix != expected_prefix:
            raise JobTransitionError("Result prefix does not match the active attempt.")
        completed = self._transition(
            job,
            status="succeeded",
            result_prefix=result_prefix,
            completed_at=_format_time(now),
            lease_until=None,
        )
        return self._swap(job, completed, "completion")

    def fail(self, job_id: str, worker_id: str, error_code: str) -> JobRecord:
        now = self._now()
        job = self._required(job_id)
        self._require_active_worker(job, worker_id, now)
        terminal = job.cancel_requested or job.attempt >= job.max_attempts
        failed = self._transition(
            job,
            status="cancelled" if job.cancel_requested else ("failed" if terminal else "queued"),
            error_code=error_code,
            worker_id=None if not terminal else worker_id,
            lease_until=None,
            completed_at=_format_time(now) if terminal else None,
        )
        return self._swap(job, failed, "failure")

    def request_cancel(self, owner_id: str, job_id: str) -> JobRecord | None:
        now = self._now()
        for _ in range(3):
            job = self._store.get(job_id)
            if job is None or job.owner_id != owner_id:
                return None
            if job.status in {"succeeded", "failed", "cancelled", "expired"}:
                return job
            if job.status == "queued":
                updated = self._transition(job, status="cancelled", cancel_requested=True, completed_at=_format_time(now))
            else:
                updated = self._transition(job, cancel_requested=True)
            if self._store.compare_and_swap(job.version, updated):
                return updated
        raise JobTransitionError("Job changed concurrently while cancelling.")

    def expire_timed_out(self, job_id: str) -> JobRecord | None:
        now = self._now()
        for _ in range(3):
            job = self._required(job_id)
            if job.status not in {"queued", "running"} or not self._is_timed_out(job, now):
                return None
            failed = self._transition(
                job,
                status="failed",
                error_code="execution_timeout",
                completed_at=_format_time(now),
                lease_until=None,
            )
            if self._store.compare_and_swap(job.version, failed):
                return failed
        raise JobTransitionError("Job changed concurrently while applying timeout.")

    def _required(self, job_id: str) -> JobRecord:
        job = self._store.get(job_id)
        if job is None:
            raise JobTransitionError("Job does not exist.")
        return job

    def _require_active_worker(self, job: JobRecord, worker_id: str, now: datetime) -> None:
        if job.status != "running" or job.worker_id != worker_id:
            raise JobTransitionError("Worker does not own the active job lease.")
        if _parse_time(job.lease_until) <= now:
            raise JobTransitionError("Worker lease has expired.")

    def _transition(self, job: JobRecord, **changes: object) -> JobRecord:
        return replace(
            job,
            **changes,
            updated_at=_format_time(self._now()),
            version=job.version + 1,
        )

    def _swap(self, previous: JobRecord, replacement: JobRecord, action: str) -> JobRecord:
        if not self._store.compare_and_swap(previous.version, replacement):
            raise JobTransitionError(f"Job changed concurrently during {action}.")
        return replacement

    def _now(self) -> datetime:
        return self._clock().astimezone(timezone.utc)

    def _is_timed_out(self, job: JobRecord, now: datetime) -> bool:
        if job.execution_started_at is None:
            return False
        return now >= _parse_time(job.execution_started_at) + timedelta(seconds=self._execution_timeout_seconds)


def artifact_prefix(job: JobRecord) -> str:
    if job.attempt < 1:
        raise JobTransitionError("Job has no active attempt.")
    return f"artifacts/{job.owner_id}/{job.job_id}/attempt-{job.attempt}/"


def _format_time(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_time(value: str | None) -> datetime:
    if value is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(value.replace("Z", "+00:00"))
