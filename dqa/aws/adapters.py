"""Small boto3-compatible adapters for the production web contracts."""

from __future__ import annotations

from dataclasses import fields
from datetime import datetime, timedelta, timezone
import json
from typing import Any

from dqa.web.artifacts import StoredArtifact
from dqa.web.jobs import JobRecord, SecurityEvent
from dqa.web.uploads import PresignedPost


_ACTIVE = {"queued", "running"}


def _job_item(job: JobRecord) -> dict[str, object]:
    item: dict[str, object] = {"pk": f"JOB#{job.job_id}", "kind": "job"}
    for field in fields(job):
        value = getattr(job, field.name)
        if value is not None:
            item[field.name] = value
    completed = job.completed_at or job.created_at
    expires = datetime.fromisoformat(completed.replace("Z", "+00:00")) + timedelta(days=30)
    item["ttl"] = int(expires.timestamp())
    return item


def _job_from_item(item: dict[str, object] | None) -> JobRecord | None:
    if not item:
        return None
    values: dict[str, object] = {}
    for field in fields(JobRecord):
        if field.name in item:
            value = item[field.name]
            if field.name in {"attempt", "max_attempts", "version"}:
                value = int(value)  # DynamoDB resources return Decimal.
            values[field.name] = value
    return JobRecord(**values)  # type: ignore[arg-type]


def _error_code(exc: Exception) -> str | None:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return None
    error = response.get("Error")
    return error.get("Code") if isinstance(error, dict) else None


class DynamoJobStore:
    """Single-table job store with transactional owner quotas and CAS updates."""

    def __init__(self, table: Any, *, max_running_per_owner: int = 1) -> None:
        self._table = table
        self._client = table.meta.client
        self._table_name = table.name
        self._max_running = max_running_per_owner

    def create_or_get_within_quota(
        self,
        job: JobRecord,
        *,
        idempotency_key: str | None,
        max_queued: int,
        max_running: int,
    ) -> JobRecord | None:
        replay = self._idempotent_job(job.owner_id, idempotency_key)
        if replay is not None:
            return replay

        owner_key = f"OWNER#{job.owner_id}"
        values = {
            ":one": 1,
            ":max_queued": max_queued,
            ":max_running": max_running,
        }
        operations: list[dict[str, object]] = [
            {
                "Update": {
                    "TableName": self._table_name,
                    "Key": {"pk": owner_key},
                    "UpdateExpression": (
                        "SET #kind = :owner_kind ADD queued_count :one"
                    ),
                    "ConditionExpression": (
                        "(attribute_not_exists(queued_count) OR queued_count < :max_queued) "
                        "AND (attribute_not_exists(running_count) OR running_count <= :max_running)"
                    ),
                    "ExpressionAttributeNames": {"#kind": "kind"},
                    "ExpressionAttributeValues": {**values, ":owner_kind": "owner"},
                }
            },
            {
                "Put": {
                    "TableName": self._table_name,
                    "Item": _job_item(job),
                    "ConditionExpression": "attribute_not_exists(pk)",
                }
            },
        ]
        if idempotency_key is not None:
            operations.append(
                {
                    "Put": {
                        "TableName": self._table_name,
                        "Item": {
                            "pk": f"IDEMP#{job.owner_id}#{idempotency_key}",
                            "kind": "idempotency",
                            "job_id": job.job_id,
                            "ttl": int((datetime.now(timezone.utc) + timedelta(days=30)).timestamp()),
                        },
                        "ConditionExpression": "attribute_not_exists(pk)",
                    }
                }
            )
        try:
            self._client.transact_write_items(TransactItems=operations)
        except Exception as exc:
            if _error_code(exc) != "TransactionCanceledException":
                raise
            return self._idempotent_job(job.owner_id, idempotency_key)
        return job

    def get(self, job_id: str) -> JobRecord | None:
        response = self._table.get_item(Key={"pk": f"JOB#{job_id}"}, ConsistentRead=True)
        return _job_from_item(response.get("Item"))

    def list_owned(self, owner_id: str, *, limit: int) -> list[JobRecord]:
        response = self._table.query(
            IndexName="owner-created-index",
            KeyConditionExpression="#owner = :owner",
            ExpressionAttributeNames={"#owner": "owner_id"},
            ExpressionAttributeValues={":owner": owner_id},
            ScanIndexForward=False,
            Limit=limit,
        )
        return [job for item in response.get("Items", []) if (job := _job_from_item(item)) is not None]

    def replace(self, job: JobRecord) -> None:
        current = self.get(job.job_id)
        if current is None:
            raise RuntimeError("Cannot replace a missing job.")
        if not self.compare_and_swap(current.version, job):
            raise RuntimeError("Job changed concurrently while replacing it.")

    def compare_and_swap(self, expected_version: int, replacement: JobRecord) -> bool:
        current = self.get(replacement.job_id)
        if current is None or current.version != expected_version or current.owner_id != replacement.owner_id:
            return False
        queued_delta = int(replacement.status == "queued") - int(current.status == "queued")
        running_delta = int(replacement.status == "running") - int(current.status == "running")
        operations: list[dict[str, object]] = [
            {
                "Put": {
                    "TableName": self._table_name,
                    "Item": _job_item(replacement),
                    "ConditionExpression": "#version = :expected",
                    "ExpressionAttributeNames": {"#version": "version"},
                    "ExpressionAttributeValues": {":expected": expected_version},
                }
            }
        ]
        if queued_delta or running_delta:
            updates: list[str] = []
            values: dict[str, object] = {}
            if queued_delta:
                updates.append("queued_count :queued")
                values[":queued"] = queued_delta
            if running_delta:
                updates.append("running_count :running")
                values[":running"] = running_delta
            operations.append(
                {
                    "Update": {
                        "TableName": self._table_name,
                        "Key": {"pk": f"OWNER#{replacement.owner_id}"},
                        "UpdateExpression": f"ADD {', '.join(updates)}",
                        "ExpressionAttributeValues": values,
                    }
                }
            )
            if running_delta > 0:
                update = operations[-1]["Update"]  # type: ignore[index]
                update["ConditionExpression"] = "attribute_not_exists(running_count) OR running_count < :max_running"  # type: ignore[index]
                update["ExpressionAttributeValues"][":max_running"] = self._max_running  # type: ignore[index]
        try:
            self._client.transact_write_items(TransactItems=operations)
        except Exception as exc:
            if _error_code(exc) == "TransactionCanceledException":
                return False
            raise
        return True

    def _idempotent_job(self, owner_id: str, key: str | None) -> JobRecord | None:
        if key is None:
            return None
        response = self._table.get_item(
            Key={"pk": f"IDEMP#{owner_id}#{key}"},
            ConsistentRead=True,
        )
        item = response.get("Item")
        if not item or not isinstance(item.get("job_id"), str):
            return None
        return self.get(item["job_id"])


class DynamoRateLimitCounter:
    def __init__(self, table: Any, *, ttl_seconds: int = 120) -> None:
        self._table = table
        self._ttl_seconds = ttl_seconds

    def consume(self, key: str, *, window_start: int, limit: int) -> bool:
        try:
            self._table.update_item(
                Key={"pk": f"RATE#{window_start}#{key}"},
                UpdateExpression="SET #kind = :kind, #ttl = :ttl ADD #count :one",
                ConditionExpression="attribute_not_exists(#count) OR #count < :limit",
                ExpressionAttributeNames={"#kind": "kind", "#ttl": "ttl", "#count": "count"},
                ExpressionAttributeValues={
                    ":kind": "rate",
                    ":ttl": window_start + self._ttl_seconds,
                    ":one": 1,
                    ":limit": limit,
                },
            )
        except Exception as exc:
            if _error_code(exc) == "ConditionalCheckFailedException":
                return False
            raise
        return True


class DynamoAdmissionGate:
    """Fail-closed workload switch changed by the cost guard or an operator."""

    def __init__(self, table: Any) -> None:
        self._table = table

    def is_open(self) -> bool:
        response = self._table.get_item(Key={"pk": "CONFIG#admission"}, ConsistentRead=True)
        item = response.get("Item")
        return bool(item and item.get("enabled") is True)


class BatchJobQueue:
    def __init__(self, client: Any, *, queue_arn: str, definition_arn: str, admission: DynamoAdmissionGate) -> None:
        self._client = client
        self._queue_arn = queue_arn
        self._definition_arn = definition_arn
        self._admission = admission

    def submit(self, job: JobRecord) -> None:
        if not self._admission.is_open():
            raise RuntimeError("New audit workloads are disabled by the cost guard.")
        self._client.submit_job(
            jobName=f"dqa-{job.job_id[:32]}",
            jobQueue=self._queue_arn,
            jobDefinition=self._definition_arn,
            parameters={"job_id": job.job_id},
            tags={"JobId": job.job_id, "OwnerId": job.owner_id},
        )


class S3UploadPostSigner:
    def __init__(self, client: Any, *, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    def create_post(
        self,
        *,
        object_key: str,
        content_length: int,
        checksum_sha256: str,
        expires_in_seconds: int,
    ) -> PresignedPost:
        response = self._client.generate_presigned_post(
            Bucket=self._bucket,
            Key=object_key,
            Fields={
                "x-amz-checksum-sha256": checksum_sha256,
                "x-amz-server-side-encryption": "AES256",
            },
            Conditions=[
                {"key": object_key},
                ["content-length-range", content_length, content_length],
                {"x-amz-checksum-sha256": checksum_sha256},
                {"x-amz-server-side-encryption": "AES256"},
            ],
            ExpiresIn=expires_in_seconds,
        )
        return PresignedPost(url=response["url"], fields=dict(response["fields"]))


class S3JobObjectStore:
    def __init__(self, client: Any, *, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    def list_objects(self, prefix: str) -> list[StoredArtifact]:
        response = self._client.list_objects_v2(Bucket=self._bucket, Prefix=prefix, MaxKeys=20)
        return [
            StoredArtifact(key=item["Key"], size=int(item["Size"]))
            for item in response.get("Contents", [])
            if isinstance(item.get("Key"), str)
        ]

    def presign_download(self, key: str, *, expires_in_seconds: int) -> str:
        return str(
            self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": key},
                ExpiresIn=expires_in_seconds,
            )
        )

    def delete_object(self, key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=key)


class JsonSecurityEventSink:
    def emit(self, event: SecurityEvent) -> None:
        print(json.dumps({"event": "security", **event.__dict__}, separators=(",", ":"), sort_keys=True))
