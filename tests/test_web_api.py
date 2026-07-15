from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import get_args, get_type_hints

from dqa.web.api import handle_request
from dqa.web.jobs import JobRecord, JobRequest, JobService, SecurityEvent
from dqa.web.lifecycle import JobLifecycle
from dqa.web.artifacts import JobArtifactService, StoredArtifact


class MemoryStore:
    def __init__(self) -> None:
        self.jobs: dict[str, JobRecord] = {}

    def create_or_get_within_quota(
        self,
        job: JobRecord,
        *,
        idempotency_key: str | None,
        max_queued: int,
        max_running: int,
    ) -> JobRecord | None:
        if idempotency_key is not None:
            existing = next(
                (
                    item
                    for item in self.jobs.values()
                    if item.owner_id == job.owner_id and item.idempotency_key == idempotency_key
                ),
                None,
            )
            if existing is not None:
                return existing
        if job.job_id in self.jobs:
            raise RuntimeError("duplicate")
        owned = [item for item in self.jobs.values() if item.owner_id == job.owner_id]
        if sum(item.status == "queued" for item in owned) >= max_queued:
            return None
        if sum(item.status == "running" for item in owned) >= max_running:
            return None
        self.jobs[job.job_id] = job
        return job

    def get(self, job_id: str) -> JobRecord | None:
        return self.jobs.get(job_id)

    def list_owned(self, owner_id: str, *, limit: int) -> list[JobRecord]:
        return sorted(
            (job for job in self.jobs.values() if job.owner_id == owner_id),
            key=lambda job: job.created_at,
            reverse=True,
        )[:limit]

    def replace(self, job: JobRecord) -> None:
        self.jobs[job.job_id] = job

    def compare_and_swap(self, expected_version: int, replacement: JobRecord) -> bool:
        current = self.jobs.get(replacement.job_id)
        if current is None or current.version != expected_version:
            return False
        self.jobs[replacement.job_id] = replacement
        return True


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


class MemoryObjects:
    def __init__(self) -> None:
        self.items: list[StoredArtifact] = []
        self.deleted: list[str] = []

    def list_objects(self, prefix: str) -> list[StoredArtifact]:
        return [item for item in self.items if item.key.startswith(prefix)]

    def presign_download(self, key: str, *, expires_in_seconds: int) -> str:
        return f"https://download.example/{key}?ttl={expires_in_seconds}"

    def delete_object(self, key: str) -> None:
        self.deleted.append(key)


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
    scope: str = "dqa/jobs",
    body: dict | str | None = None,
) -> dict:
    claims = {"sub": owner, "scope": scope} if owner else {}
    raw_body = json.dumps(body) if isinstance(body, dict) else body
    return {
        "rawPath": path,
        "body": raw_body,
        "headers": {"Idempotency-Key": "request-0001"},
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


def test_list_jobs_is_owner_scoped_and_filters_completed_comparison_inputs() -> None:
    service, store, _, _ = _service()
    first = service.submit("user-1", JobRequest(dataset_key="uploads/user-1/first.zip"))
    store.jobs[first.job_id] = replace(first, status="succeeded")
    other = replace(first, job_id="job-other", owner_id="user-2", dataset_key="uploads/user-2/other.zip")
    store.jobs[other.job_id] = other

    response = handle_request(
        {**_event("GET", "/jobs"), "queryStringParameters": {"status": "succeeded", "limit": "20"}},
        service,
        TestRateLimiter(),
    )

    assert response["statusCode"] == 200
    assert [item["job_id"] for item in _body(response)["jobs"]] == [first.job_id]


def test_owner_can_list_artifacts_without_receiving_storage_keys() -> None:
    service, store, _, _ = _service()
    job = service.submit("user-1", JobRequest(dataset_key="uploads/user-1/data.zip"))
    completed = replace(
        job,
        status="succeeded",
        attempt=1,
        result_prefix="artifacts/user-1/job-1/attempt-1/",
    )
    store.jobs[job.job_id] = completed
    objects = MemoryObjects()
    objects.items.append(StoredArtifact(key=f"{completed.result_prefix}summary.json", size=128))

    response = handle_request(
        _event("GET", "/jobs/job-1/artifacts"),
        service,
        TestRateLimiter(),
        artifacts=JobArtifactService(objects),
    )

    artifact = _body(response)["artifacts"][0]
    assert response["statusCode"] == 200
    assert artifact["name"] == "summary.json"
    assert artifact["expires_in"] == 300
    assert "key" not in artifact


def test_source_deletion_is_owner_scoped_and_rejects_active_jobs() -> None:
    service, store, _, _ = _service()
    job = service.submit("user-1", JobRequest(dataset_key="uploads/user-1/data.zip"))
    objects = MemoryObjects()
    artifacts = JobArtifactService(objects)

    active = handle_request(
        _event("DELETE", "/jobs/job-1/source"), service, TestRateLimiter(), artifacts=artifacts
    )
    store.jobs[job.job_id] = replace(job, status="failed")
    other = handle_request(
        _event("DELETE", "/jobs/job-1/source", owner="user-2"),
        service,
        TestRateLimiter(),
        artifacts=artifacts,
    )
    deleted = handle_request(
        _event("DELETE", "/jobs/job-1/source"), service, TestRateLimiter(), artifacts=artifacts
    )

    assert active["statusCode"] == 409
    assert other["statusCode"] == 404
    assert deleted["statusCode"] == 204
    assert objects.deleted == ["uploads/user-1/data.zip"]


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


def test_idempotent_replay_returns_original_without_second_enqueue() -> None:
    service, store, queue, events = _service()
    event = _event("POST", "/jobs", body={"dataset_key": "uploads/user-1/a.zip"})

    first = handle_request(event, service, TestRateLimiter())
    second = handle_request(event, service, TestRateLimiter())

    assert first["statusCode"] == 202
    assert second["statusCode"] == 202
    assert _body(first)["job"]["job_id"] == _body(second)["job"]["job_id"]
    assert len(store.jobs) == 1
    assert len(queue.submitted) == 1
    assert events.events[-1].reason == "idempotent_replay"


def test_idempotency_key_reuse_with_different_request_conflicts() -> None:
    service, _, queue, _ = _service()
    first = _event("POST", "/jobs", body={"dataset_key": "uploads/user-1/a.zip"})
    second = _event("POST", "/jobs", body={"dataset_key": "uploads/user-1/b.zip"})

    assert handle_request(first, service, TestRateLimiter())["statusCode"] == 202
    conflict = handle_request(second, service, TestRateLimiter())

    assert conflict["statusCode"] == 409
    assert len(queue.submitted) == 1


def test_job_submission_requires_idempotency_key() -> None:
    service, _, queue, _ = _service()
    event = _event("POST", "/jobs", body={"dataset_key": "uploads/user-1/a.zip"})
    event["headers"] = {}

    response = handle_request(event, service, TestRateLimiter())

    assert response["statusCode"] == 400
    assert not queue.submitted


def test_owner_can_cancel_queued_job_but_other_owner_cannot() -> None:
    service, store, _, _ = _service()
    service.submit("user-1", JobRequest(dataset_key="uploads/user-1/a.zip"))
    lifecycle = JobLifecycle(store)

    other = handle_request(
        _event("DELETE", "/jobs/job-1", owner="user-2"),
        service,
        TestRateLimiter(),
        lifecycle,
    )
    own = handle_request(
        _event("DELETE", "/jobs/job-1", owner="user-1"),
        service,
        TestRateLimiter(),
        lifecycle,
    )

    assert other["statusCode"] == 404
    assert own["statusCode"] == 202
    assert store.jobs["job-1"].status == "cancelled"


def test_job_contract_supports_all_lifecycle_states() -> None:
    states = {"queued", "running", "succeeded", "failed", "cancelled", "expired"}
    assert states == set(get_args(get_type_hints(JobRecord)["status"]))
