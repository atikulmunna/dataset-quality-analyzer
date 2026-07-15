from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .jobs import JobRecord


DOWNLOAD_TTL_SECONDS = 300


@dataclass(frozen=True)
class StoredArtifact:
    key: str
    size: int


class JobObjectStore(Protocol):
    def list_objects(self, prefix: str) -> list[StoredArtifact]: ...

    def presign_download(self, key: str, *, expires_in_seconds: int) -> str: ...

    def delete_object(self, key: str) -> None: ...


class JobArtifactService:
    """Owner-safe artifact downloads and explicit source deletion."""

    def __init__(self, objects: JobObjectStore) -> None:
        self._objects = objects

    def list_downloads(self, owner_id: str, job: JobRecord) -> list[dict[str, object]]:
        if job.owner_id != owner_id or job.status != "succeeded" or not job.result_prefix:
            return []
        expected = f"artifacts/{owner_id}/{job.job_id}/attempt-{job.attempt}/"
        if job.result_prefix != expected:
            raise ValueError("Job artifact prefix is invalid.")
        downloads: list[dict[str, object]] = []
        for item in self._objects.list_objects(expected):
            if not item.key.startswith(expected):
                raise ValueError("Artifact store returned an out-of-prefix object.")
            name = item.key.removeprefix(expected)
            if not name or "/" in name:
                continue
            downloads.append(
                {
                    "name": name,
                    "size": item.size,
                    "download_url": self._objects.presign_download(
                        item.key, expires_in_seconds=DOWNLOAD_TTL_SECONDS
                    ),
                    "expires_in": DOWNLOAD_TTL_SECONDS,
                }
            )
        return sorted(downloads, key=lambda item: str(item["name"]))

    def delete_source(self, owner_id: str, job: JobRecord) -> bool:
        if job.owner_id != owner_id:
            return False
        if job.status in {"queued", "running"}:
            raise ValueError("Source cannot be deleted while an audit is active.")
        expected_prefix = f"uploads/{owner_id}/"
        if not job.dataset_key.startswith(expected_prefix):
            raise ValueError("Job source key is invalid.")
        self._objects.delete_object(job.dataset_key)
        return True
