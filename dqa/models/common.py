from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Severity = Literal["critical", "high", "medium", "low"]
CheckStatus = Literal["completed", "skipped", "failed"]


@dataclass(frozen=True)
class SeverityCounts:
    critical: int = 0
    high: int = 0
    medium: int = 0
    low: int = 0


@dataclass(frozen=True)
class SplitStats:
    images: int
    labeled: int
    unlabeled: int
