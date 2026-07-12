from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass
from typing import Callable, Protocol
from uuid import uuid4


MAX_UPLOAD_BYTES = 2 * 1024**3


class UploadInputError(ValueError):
    """Raised when an upload declaration is unsafe or outside hosted limits."""


@dataclass(frozen=True)
class UploadRequest:
    filename: str
    size_bytes: int
    checksum_sha256: str


@dataclass(frozen=True)
class PresignedPost:
    url: str
    fields: dict[str, str]


@dataclass(frozen=True)
class UploadIntent:
    upload_id: str
    object_key: str
    expires_in_seconds: int
    post: PresignedPost


class UploadPostSigner(Protocol):
    def create_post(
        self,
        *,
        object_key: str,
        content_length: int,
        checksum_sha256: str,
        expires_in_seconds: int,
    ) -> PresignedPost: ...


class UploadService:
    def __init__(
        self,
        signer: UploadPostSigner,
        *,
        id_factory: Callable[[], str] | None = None,
        expires_in_seconds: int = 900,
    ) -> None:
        self._signer = signer
        self._id_factory = id_factory or (lambda: uuid4().hex)
        self._expires = expires_in_seconds

    def create_intent(self, owner_id: str, request: UploadRequest) -> UploadIntent:
        _validate_owner(owner_id)
        _validate_request(request)
        upload_id = self._id_factory()
        object_key = f"uploads/{owner_id}/{upload_id}/dataset.zip"
        post = self._signer.create_post(
            object_key=object_key,
            content_length=request.size_bytes,
            checksum_sha256=request.checksum_sha256,
            expires_in_seconds=self._expires,
        )
        return UploadIntent(
            upload_id=upload_id,
            object_key=object_key,
            expires_in_seconds=self._expires,
            post=post,
        )


def _validate_owner(owner_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9:_-]{1,200}", owner_id):
        raise UploadInputError("Authenticated owner is invalid.")


def _validate_request(request: UploadRequest) -> None:
    if request.filename != request.filename.rsplit("/", 1)[-1] or "\\" in request.filename:
        raise UploadInputError("filename must not contain a path.")
    if not request.filename.lower().endswith(".zip"):
        raise UploadInputError("Hosted alpha accepts ZIP datasets only.")
    if not 1 <= request.size_bytes <= MAX_UPLOAD_BYTES:
        raise UploadInputError("Upload size is outside the hosted limit.")
    try:
        decoded = base64.b64decode(request.checksum_sha256, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise UploadInputError("checksum_sha256 must be valid base64.") from exc
    if len(decoded) != 32:
        raise UploadInputError("checksum_sha256 must encode a SHA-256 digest.")
