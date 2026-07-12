from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .jobs import JobInputError, JobRequest, JobService


@dataclass(frozen=True)
class ApiResponse:
    status_code: int
    body: dict[str, object]

    def to_lambda(self) -> dict[str, object]:
        return {
            "statusCode": self.status_code,
            "headers": {"content-type": "application/json"},
            "body": json.dumps(self.body, separators=(",", ":")),
        }


def handle_request(event: dict[str, Any], jobs: JobService) -> dict[str, object]:
    owner_id = _owner_id(event)
    if owner_id is None:
        return ApiResponse(401, {"error": "unauthorized"}).to_lambda()

    method = str(event.get("requestContext", {}).get("http", {}).get("method", "")).upper()
    path = str(event.get("rawPath", ""))

    if method == "POST" and path == "/jobs":
        try:
            payload = _json_body(event)
            request = JobRequest(
                dataset_key=_required_string(payload, "dataset_key"),
                preset=payload.get("preset", "detection"),
                fail_on=payload.get("fail_on", "high"),
                near_duplicates=payload.get("near_duplicates", False),
            )
            job = jobs.submit(owner_id, request)
        except (JobInputError, TypeError, ValueError) as exc:
            return ApiResponse(400, {"error": "invalid_request", "message": str(exc)}).to_lambda()
        except Exception:
            return ApiResponse(503, {"error": "enqueue_unavailable"}).to_lambda()
        return ApiResponse(202, {"job": job.to_dict()}).to_lambda()

    if method == "GET" and path.startswith("/jobs/"):
        job_id = path.removeprefix("/jobs/")
        if not job_id or "/" in job_id:
            return ApiResponse(404, {"error": "not_found"}).to_lambda()
        job = jobs.get_owned(owner_id, job_id)
        if job is None:
            return ApiResponse(404, {"error": "not_found"}).to_lambda()
        return ApiResponse(200, {"job": job.to_dict()}).to_lambda()

    return ApiResponse(404, {"error": "not_found"}).to_lambda()


def _owner_id(event: dict[str, Any]) -> str | None:
    claims = event.get("requestContext", {}).get("authorizer", {}).get("jwt", {}).get("claims", {})
    subject = claims.get("sub") if isinstance(claims, dict) else None
    return subject if isinstance(subject, str) and subject else None


def _json_body(event: dict[str, Any]) -> dict[str, Any]:
    raw = event.get("body")
    if not isinstance(raw, str):
        raise JobInputError("JSON body is required.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise JobInputError("JSON body is invalid.") from exc
    if not isinstance(payload, dict):
        raise JobInputError("JSON body must be an object.")
    allowed = {"dataset_key", "preset", "fail_on", "near_duplicates"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise JobInputError(f"Unknown fields: {', '.join(unknown)}")
    return payload


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise JobInputError(f"{key} is required.")
    return value
