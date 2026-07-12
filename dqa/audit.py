from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal

from . import __version__
from .checks.bbox_sanity import run_bbox_sanity
from .checks.class_distribution import run_class_distribution
from .checks.duplicates import run_exact_duplicates
from .checks.integrity import run_integrity
from .checks.leakage import run_leakage
from .checks.near_duplicates import run_near_duplicates
from .config import ConfigError, DQAConfig, load_config
from .indexer import build_index, build_index_from_coco
from .io_yolo import load_dataset_spec
from .models import Finding
from .remote import RemoteDataError, resolve_data_yaml_source
from .report.html import write_html
from .report.json_writer import write_json


Severity = Literal["critical", "high", "medium", "low"]
_SEVERITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1}


@dataclass(frozen=True)
class AuditOptions:
    out: Path
    data: Path | None = None
    data_url: str | None = None
    data_url_format: str = "yolov11"
    roboflow_api_key: str | None = None
    use_remote_cache: bool = True
    remote_cache_ttl_hours: float | None = None
    config: Path | None = None
    splits: tuple[str, ...] = ("train", "val", "test")
    workers: int = 4
    max_images: int = 0
    near_duplicates: bool = False
    formats: tuple[str, ...] = ("html", "json")
    fail_on: Severity | None = None


@dataclass(frozen=True)
class AuditResult:
    exit_code: int
    summary: dict[str, object]
    flags: dict[str, object]
    index: dict[str, object]


def _enabled_checks(cfg: DQAConfig, include_near_duplicates: bool) -> list[str]:
    checks: list[str] = []
    for name in ["integrity", "class_distribution", "bbox_sanity", "duplicates", "leakage"]:
        if getattr(cfg.checks, name).enabled:
            checks.append(name)
    if cfg.checks.near_duplicates.enabled or include_near_duplicates:
        checks.append("near_duplicates")
    return checks


def _counts(findings: Iterable[Finding]) -> dict[str, int]:
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for finding in findings:
        counts[finding.severity] += 1
    return counts


def _fails_threshold(findings: Iterable[Finding], fail_on: Severity) -> bool:
    threshold = _SEVERITY_RANK[fail_on]
    return any(_SEVERITY_RANK[finding.severity] >= threshold for finding in findings)


def _serialize_finding(finding: Finding) -> dict[str, object]:
    payload: dict[str, object | None] = {
        "id": finding.id,
        "severity": finding.severity,
        "split": finding.split,
        "image": finding.image,
        "label": finding.label,
        "class_id": finding.class_id,
        "message": finding.message,
        "metrics": finding.metrics,
        "suggested_action": finding.suggested_action,
        "fingerprint": finding.fingerprint,
    }
    return {key: value for key, value in payload.items() if value is not None}


def _empty_check() -> dict[str, object]:
    return {"status": "skipped", "counts": {"critical": 0, "high": 0, "medium": 0, "low": 0}}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_index_cache(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def audit_dataset(options: AuditOptions) -> AuditResult:
    start_perf = time.perf_counter()
    started_at = _utc_now()
    cfg = load_config(options.config)
    fail_on: Severity = options.fail_on or cfg.fail_on

    if not 1 <= options.workers <= 32:
        raise ConfigError("workers must be between 1 and 32")
    if options.remote_cache_ttl_hours is not None and options.remote_cache_ttl_hours < 0:
        raise RemoteDataError("remote_cache_ttl_hours must be >= 0")

    options.out.mkdir(parents=True, exist_ok=True)
    source = resolve_data_yaml_source(
        options.data,
        options.data_url,
        options.out,
        data_url_format=options.data_url_format,
        roboflow_api_key=options.roboflow_api_key,
        use_remote_cache=options.use_remote_cache,
        remote_cache_ttl_hours=options.remote_cache_ttl_hours,
    )
    requested_splits = list(options.splits)
    previous_index = _load_index_cache(options.out / "index.json")
    source_suffix = source.suffix.lower()

    if source.is_file() and source_suffix in {".yaml", ".yml"}:
        spec = load_dataset_spec(source, requested_splits=requested_splits)
        index_result = build_index(
            spec,
            max_images=max(0, options.max_images),
            previous_index=previous_index,
            workers=options.workers,
        )
        dataset_ref = spec.data_yaml.as_posix()
        dataset_root = spec.root.as_posix()
        class_names = spec.names
        split_names = list(spec.splits)
    elif source.is_dir() or (source.is_file() and source_suffix == ".json"):
        index_result = build_index_from_coco(
            source,
            requested_splits=requested_splits,
            max_images=max(0, options.max_images),
            previous_index=previous_index,
            workers=options.workers,
        )
        dataset_ref = str(index_result.payload.get("data_source", source.as_posix()))
        dataset_root = str(index_result.payload.get("dataset_root", ""))
        class_names = list(index_result.payload.get("class_names", []))
        split_names = sorted(index_result.payload.get("splits", {}).keys())
    else:
        raise RemoteDataError(f"Unsupported dataset source: {source}")

    write_json(options.out / "index.json", index_result.payload)
    findings: list[Finding] = []
    checks_summary: dict[str, dict[str, object]] = {
        "integrity": _empty_check(),
        "class_distribution": _empty_check(),
        "bbox_sanity": _empty_check(),
        "duplicates": _empty_check(),
        "near_duplicates": _empty_check(),
        "leakage": _empty_check(),
    }

    if cfg.checks.integrity.enabled:
        found = run_integrity(index_result.payload, class_count=index_result.class_count)
        findings.extend(found)
        checks_summary["integrity"] = {"status": "completed", "counts": _counts(found)}
    if cfg.checks.class_distribution.enabled:
        found = run_class_distribution(
            index_result.payload,
            class_count=index_result.class_count,
            min_instances_per_class_warn=cfg.checks.class_distribution.min_instances_per_class_warn,
            max_class_share_warn=cfg.checks.class_distribution.max_class_share_warn,
            split_drift_jsd_warn=cfg.checks.class_distribution.split_drift_jsd_warn,
            split_drift_jsd_high=cfg.checks.class_distribution.split_drift_jsd_high,
        )
        findings.extend(found)
        checks_summary["class_distribution"] = {"status": "completed", "counts": _counts(found)}
    if cfg.checks.bbox_sanity.enabled:
        found = run_bbox_sanity(
            index_result.payload,
            min_box_area_ratio_warn=cfg.checks.bbox_sanity.min_box_area_ratio_warn,
            max_box_area_ratio_warn=cfg.checks.bbox_sanity.max_box_area_ratio_warn,
            max_boxes_per_image_warn=cfg.checks.bbox_sanity.max_boxes_per_image_warn,
            aspect_ratio_warn=cfg.checks.bbox_sanity.aspect_ratio_warn,
        )
        findings.extend(found)
        checks_summary["bbox_sanity"] = {"status": "completed", "counts": _counts(found)}
    if cfg.checks.duplicates.enabled:
        found = run_exact_duplicates(index_result.payload)
        findings.extend(found)
        checks_summary["duplicates"] = {"status": "completed", "counts": _counts(found)}
    if cfg.checks.near_duplicates.enabled or options.near_duplicates:
        found, reason = run_near_duplicates(
            index_result.payload,
            phash_hamming_threshold=cfg.checks.near_duplicates.phash_hamming_threshold,
            workers=options.workers,
        )
        findings.extend(found)
        checks_summary["near_duplicates"] = {
            "status": "completed" if reason is None else "skipped",
            "counts": _counts(found),
        }
    if cfg.checks.leakage.enabled:
        found = run_leakage(index_result.payload)
        findings.extend(found)
        checks_summary["leakage"] = {"status": "completed", "counts": _counts(found)}

    findings.sort(key=lambda item: (item.id, item.split or "", item.image or "", item.label or "", item.fingerprint))
    flags_payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "findings": [_serialize_finding(finding) for finding in findings],
    }
    write_json(options.out / "flags.json", flags_payload)

    finished_at = _utc_now()
    duration_sec = round(time.perf_counter() - start_perf, 3)
    total_counts = _counts(findings)
    build_failed = _fails_threshold(findings, fail_on)
    summary_payload: dict[str, object] = {
        "schema_version": "1.0.0",
        "run": {
            "run_id": datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
            "dqa_version": __version__,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_sec": duration_sec,
            "config": {
                "fail_on": fail_on,
                "enabled_checks": _enabled_checks(cfg, options.near_duplicates),
            },
        },
        "dataset": {
            "data_yaml": dataset_ref,
            "root": dataset_root,
            "splits": {
                split: {
                    "images": sum(1 for row in index_result.payload["images"] if row["split"] == split),
                    "labeled": sum(
                        1 for row in index_result.payload["images"] if row["split"] == split and row["label_exists"]
                    ),
                    "unlabeled": sum(
                        1 for row in index_result.payload["images"] if row["split"] == split and not row["label_exists"]
                    ),
                }
                for split in split_names
            },
            "classes": {"count": len(class_names), "names": class_names},
        },
        "checks": checks_summary,
        "totals": {
            "findings": len(findings),
            "by_severity": total_counts,
            "fail_threshold": fail_on,
            "build_failed": build_failed,
        },
    }
    write_json(options.out / "summary.json", summary_payload)
    if "html" in options.formats:
        write_html(options.out / "report.html", summary_payload, flags_payload)

    run_log = [
        f"started_at={started_at}",
        f"finished_at={finished_at}",
        f"duration_sec={duration_sec}",
        f"data={dataset_ref}",
        f"out={options.out.resolve().as_posix()}",
        f"findings={len(findings)}",
        f"build_failed={build_failed}",
        f"index_cache_hits={index_result.cache_hits}",
        f"index_cache_misses={index_result.cache_misses}",
        f"workers={options.workers}",
    ]
    (options.out / "run.log").write_text("\n".join(run_log) + "\n", encoding="utf-8")
    return AuditResult(
        exit_code=1 if build_failed else 0,
        summary=summary_payload,
        flags=flags_payload,
        index=index_result.payload,
    )
