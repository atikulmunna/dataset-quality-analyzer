from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import re
from typing import Callable, Literal, Protocol
from uuid import uuid4


JobStatus = Literal["queued", "running", "succeeded", "failed", "cancelled", "expired"]
Preset = Literal["detection", "segmentation", "segmentation_low_noise"]
Severity = Literal["critical", "high", "medium", "low"]


class JobInputError(ValueError):
    """Raised when a submitted job violates the public request contract."""


class JobQuotaError(RuntimeError):
    """Raised when an atomic per-owner job quota rejects a submission."""


class IdempotencyConflictError(RuntimeError):
    """Raised when an idempotency key is reused for a different request."""


@dataclass(frozen=True)
class JobRequest:
    dataset_key: str
    preset: Preset = "detection"
    fail_on: Severity = "high"
    near_duplicates: bool = False


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    owner_id: str
    status: JobStatus
    dataset_key: str
    preset: Preset
    fail_on: Severity
    near_duplicates: bool
    created_at: str
    updated_at: str
    error_code: str | None = None
    attempt: int = 0
    max_attempts: int = 3
    version: int = 0
    worker_id: str | None = None
    lease_until: str | None = None
    cancel_requested: bool = False
    result_prefix: str | None = None
    completed_at: str | None = None
    idempotency_key: str | None = None
    execution_started_at: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object | None] = {
            "job_id": self.job_id,
            "owner_id": self.owner_id,
            "status": self.status,
            "dataset_key": self.dataset_key,
            "preset": self.preset,
            "fail_on": self.fail_on,
            "near_duplicates": self.near_duplicates,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error_code": self.error_code,
            "cancel_requested": self.cancel_requested,
            "result_prefix": self.result_prefix,
            "completed_at": self.completed_at,
        }
        return {key: value for key, value in payload.items() if value is not None}


@dataclass(frozen=True)
class SecurityEvent:
    action: str
    outcome: Literal["allowed", "denied", "failed"]
    occurred_at: str
    owner_id: str | None = None
    job_id: str | None = None
    reason: str | None = None


class JobStore(Protocol):
    def create_or_get_within_quota(
        self,
        job: JobRecord,
        *,
        idempotency_key: str | None,
        max_queued: int,
        max_running: int,
    ) -> JobRecord | None: ...

    def get(self, job_id: str) -> JobRecord | None: ...

    def replace(self, job: JobRecord) -> None: ...


class JobQueue(Protocol):
    def submit(self, job: JobRecord) -> None: ...


class SecurityEventSink(Protocol):
    def emit(self, event: SecurityEvent) -> None: ...


class JobService:
    def __init__(
        self,
        store: JobStore,
        queue: JobQueue,
        *,
        id_factory: Callable[[], str] | None = None,
        clock: Callable[[], datetime] | None = None,
        security_events: SecurityEventSink | None = None,
        max_queued_per_owner: int = 1,
        max_running_per_owner: int = 1,
    ) -> None:
        self._store = store
        self._queue = queue
        self._id_factory = id_factory or (lambda: uuid4().hex)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._security_events = security_events
        self._max_queued = max_queued_per_owner
        self._max_running = max_running_per_owner

    def submit(self, owner_id: str, request: JobRequest, *, idempotency_key: str | None = None) -> JobRecord:
        self._validate(owner_id, request)
        if idempotency_key is not None and not re.fullmatch(r"[A-Za-z0-9._:-]{8,128}", idempotency_key):
            raise JobInputError("Idempotency-Key is invalid.")
        now = self._clock().astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        job = JobRecord(
            job_id=self._id_factory(),
            owner_id=owner_id,
            status="queued",
            dataset_key=request.dataset_key,
            preset=request.preset,
            fail_on=request.fail_on,
            near_duplicates=request.near_duplicates,
            created_at=now,
            updated_at=now,
            idempotency_key=idempotency_key,
        )
        stored = self._store.create_or_get_within_quota(
            job,
            idempotency_key=idempotency_key,
            max_queued=self._max_queued,
            max_running=self._max_running,
        )
        if stored is None:
            self.record_event("job.submit", "denied", owner_id=owner_id, reason="quota_exceeded")
            raise JobQuotaError("Active job quota exceeded.")
        if stored.job_id != job.job_id:
            if (
                stored.dataset_key != request.dataset_key
                or stored.preset != request.preset
                or stored.fail_on != request.fail_on
                or stored.near_duplicates != request.near_duplicates
            ):
                self.record_event("job.submit", "denied", owner_id=owner_id, reason="idempotency_conflict")
                raise IdempotencyConflictError("Idempotency-Key was already used for a different request.")
            self.record_event("job.submit", "allowed", owner_id=owner_id, job_id=stored.job_id, reason="idempotent_replay")
            return stored
        try:
            self._queue.submit(job)
        except Exception:
            failed = replace(job, status="failed", error_code="enqueue_failed")
            self._store.replace(failed)
            self.record_event("job.submit", "failed", owner_id=owner_id, job_id=job.job_id, reason="enqueue_failed")
            raise
        self.record_event("job.submit", "allowed", owner_id=owner_id, job_id=job.job_id)
        return job

    def get_owned(self, owner_id: str, job_id: str) -> JobRecord | None:
        job = self._store.get(job_id)
        if job is None or job.owner_id != owner_id:
            self.record_event("job.read", "denied", owner_id=owner_id, job_id=job_id, reason="not_found_or_unowned")
            return None
        self.record_event("job.read", "allowed", owner_id=owner_id, job_id=job_id)
        return job

    def record_event(
        self,
        action: str,
        outcome: Literal["allowed", "denied", "failed"],
        *,
        owner_id: str | None = None,
        job_id: str | None = None,
        reason: str | None = None,
    ) -> None:
        if self._security_events is None:
            return
        now = self._clock().astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        self._security_events.emit(
            SecurityEvent(
                action=action,
                outcome=outcome,
                occurred_at=now,
                owner_id=owner_id,
                job_id=job_id,
                reason=reason,
            )
        )

    @staticmethod
    def _validate(owner_id: str, request: JobRequest) -> None:
        if not re.fullmatch(r"[A-Za-z0-9:_-]{1,200}", owner_id):
            raise JobInputError("Authenticated owner is invalid.")
        prefix = f"uploads/{owner_id}/"
        if not request.dataset_key.startswith(prefix):
            raise JobInputError("dataset_key must belong to the authenticated owner.")
        remainder = request.dataset_key[len(prefix):]
        if not remainder or remainder.startswith("/") or ".." in remainder.split("/"):
            raise JobInputError("dataset_key is invalid.")
        if not request.dataset_key.lower().endswith(".zip"):
            raise JobInputError("Hosted alpha accepts ZIP datasets only.")
        if request.preset not in {"detection", "segmentation", "segmentation_low_noise"}:
            raise JobInputError("preset is invalid.")
        if request.fail_on not in {"critical", "high", "medium", "low"}:
            raise JobInputError("fail_on is invalid.")
        if not isinstance(request.near_duplicates, bool):
            raise JobInputError("near_duplicates must be a boolean.")
