from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .jobs import JobRecord


@dataclass(frozen=True)
class RetentionPolicy:
    source_hours_after_completion: int = 24
    successful_artifact_days: int = 7
    failed_artifact_hours: int = 48
    job_metadata_days: int = 30


@dataclass(frozen=True)
class RetentionDeadlines:
    source_expires_at: str
    artifacts_expire_at: str
    metadata_expires_at: str


def retention_deadlines(job: JobRecord, policy: RetentionPolicy | None = None) -> RetentionDeadlines:
    policy = policy or RetentionPolicy()
    if job.status not in {"succeeded", "failed", "cancelled"} or job.completed_at is None:
        raise ValueError("Retention deadlines require a terminal completed job.")
    completed = datetime.fromisoformat(job.completed_at.replace("Z", "+00:00")).astimezone(timezone.utc)
    artifact_delta = (
        timedelta(days=policy.successful_artifact_days)
        if job.status == "succeeded"
        else timedelta(hours=policy.failed_artifact_hours)
    )
    return RetentionDeadlines(
        source_expires_at=_format(completed + timedelta(hours=policy.source_hours_after_completion)),
        artifacts_expire_at=_format(completed + artifact_delta),
        metadata_expires_at=_format(completed + timedelta(days=policy.job_metadata_days)),
    )


def _format(value: datetime) -> str:
    return value.isoformat(timespec="seconds").replace("+00:00", "Z")
