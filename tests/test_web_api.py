from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import get_args, get_type_hints

from dqa.web.api import handle_request
from dqa.web.jobs import JobRecord, JobRequest, JobService, SecurityEvent


class MemoryStore:
    def __init__(self) -> None:
        self.jobs: dict[str, JobRecord] = {}

    def create_if_within_quota(self, job: JobRecord, *, max_queued: int, max_running: int) -> bool:
        if job.job_id in self.jobs:
            raise RuntimeError("duplicate")
        owned = [item for item in self.jobs.values() if item.owner_id == job.owner_id]
        if sum(item.status == "queued" for item in owned) >= max_queued:
            return False
        if sum(item.status == "running" for item in owned) >= max_running:
            return False
        self.jobs[job.job_id] = job
        return True

    def get(self, job_id: str) -> JobRecord | None:
        return self.jobs.get(job_id)

    def replace(self, job: JobRecord) -> None:
        self.jobs[job.job_id] = job


class RecordingQueue:
    def __init__(self, *, fail: bool = False) -> None:
        self.submitted: list[JobRecord] = []
        self.fail = fail

    def submit(self, job: JobRecord) -> None:
        if self.fail:
            raise RuntimeError("queue unavailable")
        self.submitted.append(job)


class EventLog:
    def __init__(self) -> None:
        self.events: list[SecurityEvent] = []

    def emit(self, event: SecurityEvent) -> None:
        self.events.append(event)


class TestRateLimiter:
    __test__ = False

    def __init__(self, allowed: bool = True) -> None:
        self.allowed = allowed
        self.calls: list[tuple[str, str]] = []

    def allow(self, owner_id: str, action: str) -> bool:
        self.calls.append((owner_id, action))
        return self.allowed


def _service(*, queue_failure: bool = False) -> tuple[JobService, MemoryStore, RecordingQueue, EventLog]:
    store = MemoryStore()
    queue = RecordingQueue(fail=queue_failure)
    events = EventLog()
    service = JobService(
        store,
        queue,
        id_factory=lambda: f"job-{len(store.jobs) + 1}",
        clock=lambda: datetime(2026, 7, 12, tzinfo=timezone.utc),
        security_events=events,
    )
    return service, store, queue, events


def _event(
    method: str,
    path: str,
    *,
    owner: str | None = "user-1",
    scope: str = "dqa:jobs",
    body: dict | str | None = None,
) -> dict:
    claims = {"sub": owner, "scope": scope} if owner else {}
    raw_body = json.dumps(body) if isinstance(body, dict) else body
    return {
        "rawPath": path,
        "body": raw_body,
        "requestContext": {
            "http": {"method": method},
            "authorizer": {"jwt": {"claims": claims}},
        },
    }


def _body(response: dict[str, object]) -> dict:
    return json.loads(str(response["body"]))


def test_submit_records_and_enqueues_without_running_audit() -> None:
    service, store, queue, events = _service()

    response = handle_request(
        _event("POST", "/jobs", body={"dataset_key": "uploads/user-1/dataset.zip"}),
        service,
        TestRateLimiter(),
    )

    assert response["statusCode"] == 202
    assert store.jobs["job-1"].status == "queued"
    assert [job.job_id for job in queue.submitted] == ["job-1"]
    assert _body(response)["job"]["owner_id"] == "user-1"
    assert events.events[-1].action == "job.submit"
    assert events.events[-1].outcome == "allowed"


def test_status_is_owner_scoped_and_hides_cross_user_job() -> None:
    service, _, _, events = _service()
    service.submit("user-1", JobRequest(dataset_key="uploads/user-1/dataset.zip"))

    own = handle_request(_event("GET", "/jobs/job-1", owner="user-1"), service, TestRateLimiter())
    other = handle_request(_event("GET", "/jobs/job-1", owner="user-2"), service, TestRateLimiter())

    assert own["statusCode"] == 200
    assert other["statusCode"] == 404
    assert events.events[-1].reason == "not_found_or_unowned"


def test_api_rejects_unauthenticated_and_unowned_inputs() -> None:
    service, _, queue, _ = _service()

    unauthorized = handle_request(_event("POST", "/jobs", owner=None, body={}), service, TestRateLimiter())
    unowned = handle_request(
        _event("POST", "/jobs", body={"dataset_key": "uploads/user-2/dataset.zip"}),
        service,
        TestRateLimiter(),
    )
    filesystem = handle_request(
        _event("POST", "/jobs", body={"dataset_key": "C:/datasets/private.zip"}),
        service,
        TestRateLimiter(),
    )

    assert unauthorized["statusCode"] == 401
    assert unowned["statusCode"] == 400
    assert filesystem["statusCode"] == 400
    assert not queue.submitted


def test_api_rejects_unknown_fields_and_invalid_types() -> None:
    service, _, _, _ = _service()

    unknown = handle_request(
        _event("POST", "/jobs", body={"dataset_key": "uploads/user-1/a.zip", "command": "rm"}),
        service,
        TestRateLimiter(),
    )
    invalid_bool = handle_request(
        _event(
            "POST",
            "/jobs",
            body={"dataset_key": "uploads/user-1/a.zip", "near_duplicates": "yes"},
        ),
        service,
        TestRateLimiter(),
    )

    assert unknown["statusCode"] == 400
    assert invalid_bool["statusCode"] == 400


def test_enqueue_failure_marks_job_failed_and_returns_service_unavailable() -> None:
    service, store, _, _ = _service(queue_failure=True)

    response = handle_request(
        _event("POST", "/jobs", body={"dataset_key": "uploads/user-1/dataset.zip"}),
        service,
        TestRateLimiter(),
    )

    assert response["statusCode"] == 503
    assert store.jobs["job-1"].status == "failed"
    assert store.jobs["job-1"].error_code == "enqueue_failed"


def test_missing_scope_is_forbidden_and_security_logged() -> None:
    service, _, queue, events = _service()

    response = handle_request(
        _event("POST", "/jobs", scope="openid", body={"dataset_key": "uploads/user-1/a.zip"}),
        service,
        TestRateLimiter(),
    )

    assert response["statusCode"] == 403
    assert not queue.submitted
    assert events.events[-1].action == "request.authorize"
    assert events.events[-1].reason == "missing_scope"


def test_rate_limit_fails_before_job_creation() -> None:
    service, store, queue, events = _service()

    response = handle_request(
        _event("POST", "/jobs", body={"dataset_key": "uploads/user-1/a.zip"}),
        service,
        TestRateLimiter(allowed=False),
    )

    assert response["statusCode"] == 429
    assert not store.jobs
    assert not queue.submitted
    assert events.events[-1].reason == "rate_limited"


def test_atomic_owner_queue_quota_rejects_second_active_job() -> None:
    service, store, queue, events = _service()
    first = service.submit("user-1", JobRequest(dataset_key="uploads/user-1/first.zip"))

    response = handle_request(
        _event("POST", "/jobs", body={"dataset_key": "uploads/user-1/second.zip"}),
        service,
        TestRateLimiter(),
    )

    assert first.status == "queued"
    assert response["statusCode"] == 429
    assert len(store.jobs) == 1
    assert len(queue.submitted) == 1
    assert events.events[-1].reason == "quota_exceeded"


def test_owner_subject_rejects_path_characters() -> None:
    service, store, queue, _ = _service()

    response = handle_request(
        _event("POST", "/jobs", owner="user/../two", body={"dataset_key": "uploads/user/../two/a.zip"}),
        service,
        TestRateLimiter(),
    )

    assert response["statusCode"] == 400
    assert not store.jobs
    assert not queue.submitted


def test_job_contract_supports_all_lifecycle_states() -> None:
    states = {"queued", "running", "succeeded", "failed", "cancelled", "expired"}
    assert states == set(get_args(get_type_hints(JobRecord)["status"]))
