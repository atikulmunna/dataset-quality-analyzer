from __future__ import annotations

from dataclasses import dataclass
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
