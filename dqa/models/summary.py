from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .common import CheckStatus, Severity, SeverityCounts, SplitStats


@dataclass(frozen=True)
class RunConfig:
    fail_on: Severity
    enabled_checks: list[str]


@dataclass(frozen=True)
class RunInfo:
    run_id: str
    dqa_version: str
    started_at: str
    finished_at: str
    duration_sec: float
    config: RunConfig


@dataclass(frozen=True)
class ClassInfo:
    count: int
    names: list[str]


@dataclass(frozen=True)
class DatasetInfo:
    data_yaml: str
    root: str
    splits: dict[str, SplitStats]
    classes: ClassInfo


@dataclass(frozen=True)
class CheckSummary:
    status: CheckStatus
    counts: SeverityCounts


@dataclass(frozen=True)
class Totals:
    findings: int
    by_severity: SeverityCounts
    fail_threshold: Severity
    build_failed: bool


@dataclass(frozen=True)
class SummaryArtifact:
    schema_version: str
    run: RunInfo
    dataset: DatasetInfo
    checks: dict[str, CheckSummary]
    totals: Totals

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
