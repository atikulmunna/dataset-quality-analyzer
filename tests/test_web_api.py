from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import get_args, get_type_hints

from dqa.web.api import handle_request
from dqa.web.jobs import JobRecord, JobRequest, JobService


class MemoryStore:
    def __init__(self) -> None:
        self.jobs: dict[str, JobRecord] = {}

    def create(self, job: JobRecord) -> None:
        if job.job_id in self.jobs:
            raise RuntimeError("duplicate")
        self.jobs[job.job_id] = job

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


def _service(*, queue_failure: bool = False) -> tuple[JobService, MemoryStore, RecordingQueue]:
    store = MemoryStore()
    queue = RecordingQueue(fail=queue_failure)
    service = JobService(
        store,
        queue,
        id_factory=lambda: "job-1",
        clock=lambda: datetime(2026, 7, 12, tzinfo=timezone.utc),
    )
    return service, store, queue


def _event(method: str, path: str, *, owner: str | None = "user-1", body: dict | str | None = None) -> dict:
    claims = {"sub": owner} if owner else {}
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
    service, store, queue = _service()

    response = handle_request(
        _event("POST", "/jobs", body={"dataset_key": "uploads/user-1/dataset.zip"}),
        service,
    )

    assert response["statusCode"] == 202
    assert store.jobs["job-1"].status == "queued"
    assert [job.job_id for job in queue.submitted] == ["job-1"]
    assert _body(response)["job"]["owner_id"] == "user-1"


def test_status_is_owner_scoped_and_hides_cross_user_job() -> None:
    service, _, _ = _service()
    service.submit("user-1", JobRequest(dataset_key="uploads/user-1/dataset.zip"))

    own = handle_request(_event("GET", "/jobs/job-1", owner="user-1"), service)
    other = handle_request(_event("GET", "/jobs/job-1", owner="user-2"), service)

    assert own["statusCode"] == 200
    assert other["statusCode"] == 404


def test_api_rejects_unauthenticated_and_unowned_inputs() -> None:
    service, _, queue = _service()

    unauthorized = handle_request(_event("POST", "/jobs", owner=None, body={}), service)
    unowned = handle_request(
        _event("POST", "/jobs", body={"dataset_key": "uploads/user-2/dataset.zip"}),
        service,
    )
    filesystem = handle_request(
        _event("POST", "/jobs", body={"dataset_key": "C:/datasets/private.zip"}),
        service,
    )

    assert unauthorized["statusCode"] == 401
    assert unowned["statusCode"] == 400
    assert filesystem["statusCode"] == 400
    assert not queue.submitted


def test_api_rejects_unknown_fields_and_invalid_types() -> None:
    service, _, _ = _service()

    unknown = handle_request(
        _event("POST", "/jobs", body={"dataset_key": "uploads/user-1/a.zip", "command": "rm"}),
        service,
    )
    invalid_bool = handle_request(
        _event(
            "POST",
            "/jobs",
            body={"dataset_key": "uploads/user-1/a.zip", "near_duplicates": "yes"},
        ),
        service,
    )

    assert unknown["statusCode"] == 400
    assert invalid_bool["statusCode"] == 400


def test_enqueue_failure_marks_job_failed_and_returns_service_unavailable() -> None:
    service, store, _ = _service(queue_failure=True)

    response = handle_request(
        _event("POST", "/jobs", body={"dataset_key": "uploads/user-1/dataset.zip"}),
        service,
    )

    assert response["statusCode"] == 503
    assert store.jobs["job-1"].status == "failed"
    assert store.jobs["job-1"].error_code == "enqueue_failed"


def test_job_contract_supports_all_lifecycle_states() -> None:
    states = {"queued", "running", "succeeded", "failed", "cancelled", "expired"}
    assert states == set(get_args(get_type_hints(JobRecord)["status"]))
