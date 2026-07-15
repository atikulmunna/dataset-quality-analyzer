from __future__ import annotations

from dataclasses import replace

from dqa.aws.adapters import (
    BatchJobQueue,
    DynamoAdmissionGate,
    DynamoJobStore,
    S3UploadPostSigner,
    _job_from_item,
    _job_item,
)
from dqa.aws.api_handler import handler
from dqa.web.jobs import JobRecord


def _job(**changes: object) -> JobRecord:
    base = JobRecord(
        job_id="job-1",
        owner_id="owner-1",
        status="queued",
        dataset_key="uploads/owner-1/upload/dataset.zip",
        preset="detection",
        fail_on="high",
        near_duplicates=False,
        created_at="2026-07-15T00:00:00Z",
        updated_at="2026-07-15T00:00:00Z",
        idempotency_key="request-1",
    )
    return replace(base, **changes)


class _Client:
    def __init__(self) -> None:
        self.transactions: list[list[dict[str, object]]] = []
        self.submissions: list[dict[str, object]] = []

    def transact_write_items(self, *, TransactItems: list[dict[str, object]]) -> None:
        self.transactions.append(TransactItems)

    def submit_job(self, **kwargs: object) -> None:
        self.submissions.append(kwargs)


class _Meta:
    def __init__(self, client: _Client) -> None:
        self.client = client


class _Table:
    name = "dqa-test"

    def __init__(self, items: dict[str, dict[str, object]] | None = None) -> None:
        self.items = items or {}
        self.client = _Client()
        self.meta = _Meta(self.client)

    def get_item(self, *, Key: dict[str, str], ConsistentRead: bool = False) -> dict[str, object]:
        del ConsistentRead
        item = self.items.get(Key["pk"])
        return {"Item": item} if item is not None else {}


def test_job_dynamo_round_trip_preserves_internal_fields() -> None:
    job = _job(status="running", attempt=2, version=4, worker_id="worker-1")
    item = _job_item(job)

    assert item["pk"] == "JOB#job-1"
    assert isinstance(item["ttl"], int)
    assert _job_from_item(item) == job


def test_job_store_create_is_one_transaction_with_owner_quota() -> None:
    table = _Table()
    store = DynamoJobStore(table)

    created = store.create_or_get_within_quota(
        _job(), idempotency_key="request-1", max_queued=1, max_running=1
    )

    assert created == _job()
    operations = table.client.transactions[0]
    assert len(operations) == 3
    assert "queued_count < :max_queued" in operations[0]["Update"]["ConditionExpression"]  # type: ignore[index]


def test_batch_queue_fails_closed_and_submits_parameterized_job() -> None:
    closed_table = _Table({"CONFIG#admission": {"pk": "CONFIG#admission", "enabled": False}})
    client = _Client()
    queue = BatchJobQueue(
        client,
        queue_arn="queue-arn",
        definition_arn="definition-arn",
        admission=DynamoAdmissionGate(closed_table),
    )

    try:
        queue.submit(_job())
    except RuntimeError as exc:
        assert "cost guard" in str(exc)
    else:
        raise AssertionError("closed admission gate accepted a job")

    closed_table.items["CONFIG#admission"]["enabled"] = True
    queue.submit(_job())
    assert client.submissions[0]["parameters"] == {"job_id": "job-1"}
    assert "retryStrategy" not in client.submissions[0]


def test_s3_post_signer_binds_key_length_checksum_and_encryption() -> None:
    class S3:
        request: dict[str, object]

        def generate_presigned_post(self, **kwargs: object) -> dict[str, object]:
            self.request = kwargs
            return {"url": "https://bucket.example", "fields": {"key": kwargs["Key"]}}

    client = S3()
    post = S3UploadPostSigner(client, bucket="bucket").create_post(
        object_key="uploads/owner/id/dataset.zip",
        content_length=123,
        checksum_sha256="checksum",
        expires_in_seconds=900,
    )

    assert post.url == "https://bucket.example"
    assert ["content-length-range", 123, 123] in client.request["Conditions"]  # type: ignore[operator]
    assert {"x-amz-server-side-encryption": "AES256"} in client.request["Conditions"]  # type: ignore[operator]


def test_lambda_health_does_not_initialize_aws_clients() -> None:
    response = handler({"rawPath": "/health"}, None)
    assert response["statusCode"] == 200
