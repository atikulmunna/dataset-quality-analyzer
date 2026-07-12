from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from .jobs import IdempotencyConflictError, JobInputError, JobQuotaError, JobRequest, JobService
from .lifecycle import JobLifecycle


@dataclass(frozen=True)
class AuthContext:
    owner_id: str
    scopes: frozenset[str]


class RateLimiter(Protocol):
    def allow(self, owner_id: str, action: str) -> bool: ...


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


def handle_request(
    event: dict[str, Any],
    jobs: JobService,
    rate_limiter: RateLimiter,
    lifecycle: JobLifecycle | None = None,
) -> dict[str, object]:
    auth = auth_context(event)
    if auth is None:
        jobs.record_event("request.authenticate", "denied", reason="missing_subject")
        return ApiResponse(401, {"error": "unauthorized"}).to_lambda()
    if "dqa:jobs" not in auth.scopes:
        jobs.record_event("request.authorize", "denied", owner_id=auth.owner_id, reason="missing_scope")
        return ApiResponse(403, {"error": "forbidden"}).to_lambda()

    method = str(event.get("requestContext", {}).get("http", {}).get("method", "")).upper()
    path = str(event.get("rawPath", ""))
    action = f"{method} {path.split('/', 2)[1] if path.startswith('/') and len(path.split('/')) > 1 else path}"
    if not rate_limiter.allow(auth.owner_id, action):
        jobs.record_event("request.rate_limit", "denied", owner_id=auth.owner_id, reason="rate_limited")
        return ApiResponse(429, {"error": "rate_limited"}).to_lambda()

    if method == "POST" and path == "/jobs":
        try:
            payload = _json_body(event)
            idempotency_key = _idempotency_key(event)
            request = JobRequest(
                dataset_key=_required_string(payload, "dataset_key"),
                preset=payload.get("preset", "detection"),
                fail_on=payload.get("fail_on", "high"),
                near_duplicates=payload.get("near_duplicates", False),
            )
            job = jobs.submit(auth.owner_id, request, idempotency_key=idempotency_key)
        except IdempotencyConflictError as exc:
            return ApiResponse(409, {"error": "idempotency_conflict", "message": str(exc)}).to_lambda()
        except JobQuotaError as exc:
            return ApiResponse(429, {"error": "quota_exceeded", "message": str(exc)}).to_lambda()
        except (JobInputError, TypeError, ValueError) as exc:
            return ApiResponse(400, {"error": "invalid_request", "message": str(exc)}).to_lambda()
        except Exception:
            return ApiResponse(503, {"error": "enqueue_unavailable"}).to_lambda()
        return ApiResponse(202, {"job": job.to_dict()}).to_lambda()

    if method == "GET" and path.startswith("/jobs/"):
        job_id = path.removeprefix("/jobs/")
        if not job_id or "/" in job_id:
            return ApiResponse(404, {"error": "not_found"}).to_lambda()
        job = jobs.get_owned(auth.owner_id, job_id)
        if job is None:
            return ApiResponse(404, {"error": "not_found"}).to_lambda()
        return ApiResponse(200, {"job": job.to_dict()}).to_lambda()

    if method == "DELETE" and path.startswith("/jobs/"):
        job_id = path.removeprefix("/jobs/")
        if not job_id or "/" in job_id:
            return ApiResponse(404, {"error": "not_found"}).to_lambda()
        if lifecycle is None:
            return ApiResponse(503, {"error": "cancellation_unavailable"}).to_lambda()
        job = lifecycle.request_cancel(auth.owner_id, job_id)
        if job is None:
            return ApiResponse(404, {"error": "not_found"}).to_lambda()
        return ApiResponse(202, {"job": job.to_dict()}).to_lambda()

    return ApiResponse(404, {"error": "not_found"}).to_lambda()


def auth_context(event: dict[str, Any]) -> AuthContext | None:
    claims = event.get("requestContext", {}).get("authorizer", {}).get("jwt", {}).get("claims", {})
    subject = claims.get("sub") if isinstance(claims, dict) else None
    if not isinstance(subject, str) or not subject:
        return None
    raw_scope = claims.get("scope", "")
    scopes = frozenset(raw_scope.split()) if isinstance(raw_scope, str) else frozenset()
    return AuthContext(owner_id=subject, scopes=scopes)


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


def _idempotency_key(event: dict[str, Any]) -> str:
    headers = event.get("headers", {})
    if not isinstance(headers, dict):
        raise JobInputError("Idempotency-Key header is required.")
    value = next((item for key, item in headers.items() if str(key).lower() == "idempotency-key"), None)
    if not isinstance(value, str) or not value:
        raise JobInputError("Idempotency-Key header is required.")
    return value
