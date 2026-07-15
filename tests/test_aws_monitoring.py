from __future__ import annotations

import json

import pytest

from dqa.aws.monitoring import handle_event


QUEUE = "arn:aws:batch:us-east-1:123456789012:job-queue/dqa-dev-audit"


class FakeBatch:
    def list_jobs(self, **request: object) -> dict[str, object]:
        counts = {"SUBMITTED": 1, "PENDING": 0, "RUNNABLE": 1, "STARTING": 0}
        return {"jobSummaryList": [{}] * counts[str(request["jobStatus"])]}


class FakeCloudWatch:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def put_metric_data(self, **request: object) -> None:
        self.requests.append(request)


def _failed_event() -> dict[str, object]:
    return {
        "detail": {
            "jobQueue": QUEUE,
            "jobId": "batch-123",
            "status": "FAILED",
            "startedAt": 1_000,
            "stoppedAt": 6_500,
            "parameters": {"job_id": "job-42"},
        }
    }


def test_failed_batch_event_emits_alarm_metrics_and_searchable_log(capsys: pytest.CaptureFixture[str]) -> None:
    cloudwatch = FakeCloudWatch()
    handle_event(
        _failed_event(),
        batch=FakeBatch(),
        cloudwatch=cloudwatch,
        queue_arn=QUEUE,
        namespace="DQA/dev",
        environment="dev",
    )

    metrics = {item["MetricName"]: item for item in cloudwatch.requests[0]["MetricData"]}  # type: ignore[index]
    assert metrics["QueueDepth"]["Value"] == 2
    assert metrics["JobFailures"]["Value"] == 1
    assert metrics["AuditRuntimeSeconds"]["Value"] == 5.5
    event = json.loads(capsys.readouterr().out)
    assert event["event"] == "monitor.batch_state"
    assert event["job_id"] == "job-42"
    assert event["batch_job_id"] == "batch-123"


def test_monitor_rejects_events_from_another_queue() -> None:
    event = _failed_event()
    event["detail"]["jobQueue"] = "other"  # type: ignore[index]
    with pytest.raises(ValueError, match="unexpected queue"):
        handle_event(
            event,
            batch=FakeBatch(),
            cloudwatch=FakeCloudWatch(),
            queue_arn=QUEUE,
            namespace="DQA/dev",
            environment="dev",
        )
