"""AWS Lambda entry point for authenticated job and upload routes."""

from __future__ import annotations

import os
from typing import Any

from dqa.web.api import ApiResponse, handle_request
from dqa.web.artifacts import JobArtifactService
from dqa.web.jobs import JobService
from dqa.web.lifecycle import JobLifecycle
from dqa.web.security import FixedWindowRateLimiter
from dqa.web.upload_api import handle_upload_request
from dqa.web.uploads import UploadService

from .adapters import (
    BatchJobQueue,
    DynamoAdmissionGate,
    DynamoJobStore,
    DynamoRateLimitCounter,
    JsonSecurityEventSink,
    S3UploadPostSigner,
    S3JobObjectStore,
)


_runtime: tuple[JobService, JobLifecycle, FixedWindowRateLimiter, UploadService, JobArtifactService] | None = None


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Required environment variable {name} is missing.")
    return value


def _build_runtime() -> tuple[JobService, JobLifecycle, FixedWindowRateLimiter, UploadService, JobArtifactService]:
    import boto3

    table = boto3.resource("dynamodb").Table(_required("DQA_TABLE_NAME"))
    store = DynamoJobStore(table)
    admission = DynamoAdmissionGate(table)
    queue = BatchJobQueue(
        boto3.client("batch"),
        queue_arn=_required("DQA_BATCH_QUEUE_ARN"),
        definition_arn=_required("DQA_JOB_DEFINITION_ARN"),
        admission=admission,
    )
    limiter = FixedWindowRateLimiter(DynamoRateLimitCounter(table))
    jobs = JobService(store, queue, security_events=JsonSecurityEventSink())
    lifecycle = JobLifecycle(store)
    s3 = boto3.client("s3")
    bucket = _required("DQA_DATA_BUCKET")
    uploads = UploadService(S3UploadPostSigner(s3, bucket=bucket))
    artifacts = JobArtifactService(S3JobObjectStore(s3, bucket=bucket))
    return jobs, lifecycle, limiter, uploads, artifacts


def handler(event: dict[str, Any], _context: object) -> dict[str, object]:
    global _runtime
    if event.get("rawPath") == "/health":
        return ApiResponse(200, {"status": "ok"}).to_lambda()
    if _runtime is None:
        _runtime = _build_runtime()
    jobs, lifecycle, limiter, uploads, artifacts = _runtime
    if event.get("rawPath") == "/uploads":
        return handle_upload_request(event, uploads, limiter)
    return handle_request(event, jobs, limiter, lifecycle, artifacts)
