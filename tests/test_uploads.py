from __future__ import annotations

import base64

import pytest

from dqa.web.uploads import MAX_UPLOAD_BYTES, PresignedPost, UploadInputError, UploadRequest, UploadService


_CHECKSUM = base64.b64encode(b"x" * 32).decode("ascii")


class RecordingSigner:
    def __init__(self) -> None:
        self.request: dict[str, object] | None = None

    def create_post(self, **kwargs) -> PresignedPost:
        self.request = kwargs
        return PresignedPost(url="https://uploads.example", fields={"key": str(kwargs["object_key"])})


def test_upload_intent_is_owner_scoped_checksum_bound_and_short_lived() -> None:
    signer = RecordingSigner()
    service = UploadService(signer, id_factory=lambda: "upload-1")

    intent = service.create_intent(
        "user-1",
        UploadRequest(filename="dataset.zip", size_bytes=1234, checksum_sha256=_CHECKSUM),
    )

    assert intent.object_key == "uploads/user-1/upload-1/dataset.zip"
    assert intent.expires_in_seconds == 900
    assert signer.request == {
        "object_key": intent.object_key,
        "content_length": 1234,
        "checksum_sha256": _CHECKSUM,
        "expires_in_seconds": 900,
    }


@pytest.mark.parametrize(
    ("upload_request", "message"),
    [
        (UploadRequest("../dataset.zip", 1, _CHECKSUM), "path"),
        (UploadRequest("dataset.tar", 1, _CHECKSUM), "ZIP"),
        (UploadRequest("dataset.zip", 0, _CHECKSUM), "size"),
        (UploadRequest("dataset.zip", MAX_UPLOAD_BYTES + 1, _CHECKSUM), "size"),
        (UploadRequest("dataset.zip", 1, "invalid"), "checksum"),
    ],
)
def test_upload_intent_rejects_unsafe_declarations(upload_request: UploadRequest, message: str) -> None:
    with pytest.raises(UploadInputError, match=message):
        UploadService(RecordingSigner()).create_intent("user-1", upload_request)


def test_upload_intent_rejects_unsafe_owner() -> None:
    with pytest.raises(UploadInputError, match="owner"):
        UploadService(RecordingSigner()).create_intent(
            "user/../two",
            UploadRequest("dataset.zip", 1, _CHECKSUM),
        )
