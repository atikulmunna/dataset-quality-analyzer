"""Turn AWS Batch state changes into a small bounded metric set."""

from __future__ import annotations

import os
from typing import Any

from .observability import emit_event


_QUEUED_STATUSES = ("SUBMITTED", "PENDING", "RUNNABLE", "STARTING")
_clients: tuple[Any, Any] | None = None


def _count_jobs(batch: Any, queue_arn: str) -> int:
    total = 0
    for status in _QUEUED_STATUSES:
        token: str | None = None
        while True:
            request: dict[str, object] = {
                "jobQueue": queue_arn,
                "jobStatus": status,
                "maxResults": 100,
            }
            if token:
                request["nextToken"] = token
            response = batch.list_jobs(**request)
            total += len(response.get("jobSummaryList", []))
            token = response.get("nextToken")
            if not token:
                break
    return total


def handle_event(
    event: dict[str, Any],
    *,
    batch: Any,
    cloudwatch: Any,
    queue_arn: str,
    namespace: str,
    environment: str,
) -> None:
    detail = event.get("detail")
    if not isinstance(detail, dict):
        raise ValueError("Batch event detail is missing.")
    event_queue = detail.get("jobQueue")
    if event_queue != queue_arn:
        raise ValueError("Batch event belongs to an unexpected queue.")

    status = str(detail.get("status", "UNKNOWN"))
    parameters = detail.get("parameters")
    job_id = str(parameters.get("job_id", "unknown")) if isinstance(parameters, dict) else "unknown"
    batch_job_id = str(detail.get("jobId", "unknown"))
    queued = _count_jobs(batch, queue_arn)
    dimensions = [{"Name": "Environment", "Value": environment}]
    metrics: list[dict[str, object]] = [
        {
            "MetricName": "QueueDepth",
            "Dimensions": dimensions,
            "Unit": "Count",
            "Value": queued,
        }
    ]
    if status == "FAILED":
        metrics.append(
            {
                "MetricName": "JobFailures",
                "Dimensions": dimensions,
                "Unit": "Count",
                "Value": 1,
            }
        )

    started = detail.get("startedAt")
    stopped = detail.get("stoppedAt")
    duration_seconds: float | None = None
    if status in {"SUCCEEDED", "FAILED"} and isinstance(started, (int, float)) and isinstance(stopped, (int, float)):
        duration_seconds = max(0.0, (stopped - started) / 1000.0)
        metrics.append(
            {
                "MetricName": "AuditRuntimeSeconds",
                "Dimensions": dimensions,
                "Unit": "Seconds",
                "Value": duration_seconds,
            }
        )

    cloudwatch.put_metric_data(Namespace=namespace, MetricData=metrics)
    emit_event(
        "monitor.batch_state",
        job_id=job_id,
        batch_job_id=batch_job_id,
        status=status,
        queue_depth=queued,
        duration_seconds=duration_seconds,
    )


def handler(event: dict[str, Any], _context: object) -> None:
    global _clients
    if _clients is None:
        import boto3

        _clients = (boto3.client("batch"), boto3.client("cloudwatch"))
    batch, cloudwatch = _clients
    handle_event(
        event,
        batch=batch,
        cloudwatch=cloudwatch,
        queue_arn=os.environ["DQA_BATCH_QUEUE_ARN"],
        namespace=os.environ["DQA_METRIC_NAMESPACE"],
        environment=os.environ["DQA_ENVIRONMENT"],
    )
