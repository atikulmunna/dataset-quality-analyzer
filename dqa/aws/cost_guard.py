"""Budget-notification Lambda that closes the workload admission switch."""

from __future__ import annotations

import json
import os
from typing import Any


def handler(event: dict[str, Any], _context: object) -> dict[str, object]:
    import boto3

    records = event.get("Records", [])
    if not isinstance(records, list) or not records:
        raise ValueError("A budget SNS record is required.")
    for record in records:
        message = record.get("Sns", {}).get("Message") if isinstance(record, dict) else None
        if isinstance(message, str):
            json.loads(message) if message.startswith("{") else message
    table_name = os.environ.get("DQA_TABLE_NAME")
    if not table_name:
        raise RuntimeError("DQA_TABLE_NAME is missing.")
    boto3.resource("dynamodb").Table(table_name).update_item(
        Key={"pk": "CONFIG#admission"},
        UpdateExpression="SET enabled = :disabled, reason = :reason, updated_at = :updated",
        ExpressionAttributeValues={
            ":disabled": False,
            ":reason": "aws_budget_threshold",
            ":updated": event.get("Records", [{}])[0].get("Sns", {}).get("Timestamp", "unknown"),
        },
    )
    return {"admission_enabled": False}

