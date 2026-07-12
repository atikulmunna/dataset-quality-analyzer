from __future__ import annotations

import json
from typing import Any

from .api import ApiResponse, RateLimiter, auth_context
from .uploads import UploadInputError, UploadRequest, UploadService


def handle_upload_request(
    event: dict[str, Any],
    uploads: UploadService,
    rate_limiter: RateLimiter,
) -> dict[str, object]:
    auth = auth_context(event)
    if auth is None:
        return ApiResponse(401, {"error": "unauthorized"}).to_lambda()
    if "dqa:jobs" not in auth.scopes:
        return ApiResponse(403, {"error": "forbidden"}).to_lambda()
    method = str(event.get("requestContext", {}).get("http", {}).get("method", "")).upper()
    if method != "POST" or event.get("rawPath") != "/uploads":
        return ApiResponse(404, {"error": "not_found"}).to_lambda()
    if not rate_limiter.allow(auth.owner_id, "POST uploads"):
        return ApiResponse(429, {"error": "rate_limited"}).to_lambda()

    try:
        payload = _body(event)
        intent = uploads.create_intent(
            auth.owner_id,
            UploadRequest(
                filename=_string(payload, "filename"),
                size_bytes=_integer(payload, "size_bytes"),
                checksum_sha256=_string(payload, "checksum_sha256"),
            ),
        )
    except (UploadInputError, TypeError, ValueError) as exc:
        return ApiResponse(400, {"error": "invalid_request", "message": str(exc)}).to_lambda()

    return ApiResponse(
        201,
        {
            "upload": {
                "upload_id": intent.upload_id,
                "object_key": intent.object_key,
                "expires_in_seconds": intent.expires_in_seconds,
                "post": {"url": intent.post.url, "fields": intent.post.fields},
            }
        },
    ).to_lambda()


def _body(event: dict[str, Any]) -> dict[str, Any]:
    raw = event.get("body")
    if not isinstance(raw, str):
        raise UploadInputError("JSON body is required.")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise UploadInputError("JSON body is invalid.") from exc
    if not isinstance(payload, dict):
        raise UploadInputError("JSON body must be an object.")
    allowed = {"filename", "size_bytes", "checksum_sha256"}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise UploadInputError(f"Unknown fields: {', '.join(unknown)}")
    return payload


def _string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise UploadInputError(f"{key} is required.")
    return value


def _integer(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise UploadInputError(f"{key} must be an integer.")
    return value
