from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .common import Severity


@dataclass(frozen=True)
class Finding:
    id: str
    severity: Severity
    message: str
    fingerprint: str
    split: str | None = None
    image: str | None = None
    label: str | None = None
    class_id: int | None = None
    metrics: dict[str, Any] | None = None
    suggested_action: str | None = None


@dataclass(frozen=True)
class FlagsArtifact:
    schema_version: str
    findings: list[Finding]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
