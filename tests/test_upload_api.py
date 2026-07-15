from __future__ import annotations

import base64
import json

from dqa.web.upload_api import handle_upload_request
from dqa.web.uploads import PresignedPost, UploadService


class Signer:
    def create_post(self, **kwargs) -> PresignedPost:
        return PresignedPost("https://s3.example", {"key": str(kwargs["object_key"])})


class Limiter:
    def __init__(self, allowed: bool = True) -> None:
        self.allowed = allowed

    def allow(self, owner_id: str, action: str) -> bool:
        return self.allowed and owner_id == "user-1" and action == "POST uploads"


def _event(*, scope: str = "dqa/jobs", body: dict | None = None) -> dict:
    return {
        "rawPath": "/uploads",
        "body": json.dumps(body or {}),
        "requestContext": {
            "http": {"method": "POST"},
            "authorizer": {"jwt": {"claims": {"sub": "user-1", "scope": scope}}},
        },
    }


def test_upload_endpoint_returns_direct_post_without_dataset_bytes() -> None:
    checksum = base64.b64encode(b"x" * 32).decode("ascii")
    service = UploadService(Signer(), id_factory=lambda: "upload-1")

    response = handle_upload_request(
        _event(body={"filename": "dataset.zip", "size_bytes": 100, "checksum_sha256": checksum}),
        service,
        Limiter(),
    )
    body = json.loads(str(response["body"]))

    assert response["statusCode"] == 201
    assert body["upload"]["object_key"] == "uploads/user-1/upload-1/dataset.zip"
    assert body["upload"]["post"]["url"] == "https://s3.example"


def test_upload_endpoint_requires_scope_and_rate_limit() -> None:
    service = UploadService(Signer())

    forbidden = handle_upload_request(_event(scope="openid"), service, Limiter())
    limited = handle_upload_request(_event(), service, Limiter(allowed=False))

    assert forbidden["statusCode"] == 403
    assert limited["statusCode"] == 429


def test_upload_endpoint_rejects_unknown_or_missing_fields() -> None:
    service = UploadService(Signer())

    unknown = handle_upload_request(_event(body={"command": "audit"}), service, Limiter())
    missing = handle_upload_request(_event(body={"filename": "dataset.zip"}), service, Limiter())

    assert unknown["statusCode"] == 400
    assert missing["statusCode"] == 400
