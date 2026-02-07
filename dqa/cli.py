from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError as JsonSchemaError
from jsonschema.exceptions import ValidationError as JsonValidationError

from .checks.bbox_sanity import run_bbox_sanity
from .checks.class_distribution import run_class_distribution
from .checks.duplicates import run_exact_duplicates
from .checks.integrity import run_integrity
from .checks.leakage import run_leakage
from .checks.near_duplicates import run_near_duplicates
from .config import ConfigError, load_config
from .indexer import build_index, build_index_from_coco
from .io_yolo import DatasetSpecError, load_dataset_spec
from .models import Finding
from .remote import RemoteDataError, resolve_data_yaml_source
from .report.html import write_html
from .report.json_writer import write_json


_SEVERITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1}


class ExplainError(ValueError):
    """Raised when explain input artifacts are invalid."""


class ValidateError(ValueError):
    """Raised when validate inputs are invalid."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dqa", description="Dataset Quality Analyzer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Audit a YOLO dataset")
    audit.add_argument("--data", type=Path, help="Path to local data.yaml")
    audit.add_argument("--data-url", type=str, help="Remote dataset URL (currently supports Roboflow URLs)")
    audit.add_argument("--data-url-format", type=str, default="yolov11", help="Remote export format (default: yolov11)")
    audit.add_argument("--roboflow-api-key", type=str, default=None, help="Optional Roboflow API key override")
    audit.add_argument("--no-remote-cache", action="store_true", help="Disable remote dataset cache reuse")
    audit.add_argument("--remote-cache-ttl-hours", type=float, default=None, help="Remote cache TTL in hours (default: no TTL)")
    audit.add_argument("--out", required=True, type=Path, help="Output directory")
    audit.add_argument("--config", type=Path, default=Path("dqa.yaml"), help="Path to dqa config")
    audit.add_argument("--splits", default="train,val,test", help="Comma-separated splits")
    audit.add_argument("--workers", type=int, default=8)
    audit.add_argument("--max-images", type=int, default=0)
    audit.add_argument("--near-dup", action="store_true")
    audit.add_argument("--format", default="html,json")
    audit.add_argument("--fail-on", choices=["critical", "high", "medium", "low"], default=None)

    explain = subparsers.add_parser("explain", help="Explain findings from a completed DQA run")
    explain.add_argument("--run", type=Path, default=None, help="Run directory containing summary.json and flags.json")
    explain.add_argument("--summary", type=Path, default=None, help="Path to summary.json")
    explain.add_argument("--flags", type=Path, default=None, help="Path to flags.json")

    validate = subparsers.add_parser("validate", help="Validate a DQA JSON artifact against a JSON schema")
    validate.add_argument("--artifact", required=True, type=Path, help="Path to artifact JSON (e.g., summary.json)")
    validate.add_argument("--schema", required=True, type=Path, help="Path to JSON schema")

    diff = subparsers.add_parser("diff", help="Compare two DQA runs and show finding regressions")
    diff.add_argument("--old", required=True, type=Path, help="Old run directory containing summary.json and flags.json")
    diff.add_argument("--new", required=True, type=Path, help="New run directory containing summary.json and flags.json")
    diff.add_argument("--fail-on-regression", choices=["critical", "high", "medium", "low"], default=None)

    return parser


def _enabled_checks(cfg: object, include_near_dup_flag: bool) -> list[str]:
    checks: list[str] = []
    checks_obj = getattr(cfg, "checks")
    for name in ["integrity", "class_distribution", "bbox_sanity", "duplicates", "leakage"]:
        if getattr(getattr(checks_obj, name), "enabled"):
            checks.append(name)
    if getattr(checks_obj.near_duplicates, "enabled") or include_near_dup_flag:
        checks.append("near_duplicates")
    return checks


def _counts(findings: Iterable[Finding]) -> dict[str, int]:
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for finding in findings:
        counts[finding.severity] += 1
    return counts


def _counts_from_payload(findings: list[dict]) -> dict[str, int]:
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for row in findings:
        sev = str(row.get("severity", "")).lower()
        if sev in counts:
            counts[sev] += 1
    return counts


def _fails_threshold(findings: Iterable[Finding], fail_on: str) -> bool:
    threshold = _SEVERITY_RANK[fail_on]
    for finding in findings:
        if _SEVERITY_RANK[finding.severity] >= threshold:
            return True
    return False


def _serialize_finding(finding: Finding) -> dict[str, object]:
    return {
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


def _empty_check() -> dict[str, object]:
    return {"status": "skipped", "counts": {"critical": 0, "high": 0, "medium": 0, "low": 0}}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _load_json(path: Path, expected_name: str) -> dict:
    if not path.exists():
        raise ExplainError(f"Missing {expected_name}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExplainError(f"Invalid JSON in {expected_name}: {path}") from exc
    if not isinstance(payload, dict):
        raise ExplainError(f"{expected_name} must be a JSON object: {path}")
    return payload


def _recommendation_for_id(finding_id: str) -> str:
    mapping = {
        "CLASS_LOW_SUPPORT": "Increase labeled samples for low-support classes or merge/remove unstable classes.",
        "CLASS_IMBALANCE_HIGH": "Rebalance sampling and collect more minority-class data.",
        "CLASS_SPLIT_DRIFT": "Rebuild splits to align class distributions across train/val/test.",
        "LEAKAGE_EXACT_TRAIN_VAL": "Re-split dataset to eliminate train/val overlap.",
        "LEAKAGE_EXACT_TRAIN_TEST": "Re-split dataset to eliminate train/test overlap.",
        "DUPLICATE_ACROSS_SPLITS": "Deduplicate across splits before training.",
        "INTEGRITY_MISSING_LABEL": "Add missing label files or move unlabeled images out of training split.",
    }
    return mapping.get(finding_id, "Review flagged samples and apply targeted cleanup for this issue type.")


def run_explain(args: argparse.Namespace) -> int:
    if args.run is not None and (args.summary is not None or args.flags is not None):
        raise ExplainError("Use --run OR (--summary and --flags), not both.")

    if args.run is not None:
        summary_path = args.run / "summary.json"
        flags_path = args.run / "flags.json"
        run_label = args.run.as_posix()
    else:
        if args.summary is None or args.flags is None:
            raise ExplainError("Provide --run, or provide both --summary and --flags.")
        summary_path = args.summary
        flags_path = args.flags
        run_label = f"summary={summary_path.as_posix()} flags={flags_path.as_posix()}"

    summary = _load_json(summary_path, "summary.json")
    flags = _load_json(flags_path, "flags.json")

    findings = flags.get("findings", [])
    if not isinstance(findings, list):
        raise ExplainError("flags.json must contain a 'findings' array.")

    totals = summary.get("totals", {})
    by_severity = totals.get("by_severity")
    if not isinstance(by_severity, dict):
        by_severity = _counts_from_payload([f for f in findings if isinstance(f, dict)])

    finding_rows = [f for f in findings if isinstance(f, dict)]
    id_counts = Counter(str(f.get("id", "UNKNOWN")) for f in finding_rows)

    print("DQA Explain")
    print(f"run={run_label}")
    print(f"findings={len(finding_rows)} build_failed={totals.get('build_failed')}")
    print(
        "severity: "
        f"critical={by_severity.get('critical', 0)} "
        f"high={by_severity.get('high', 0)} "
        f"medium={by_severity.get('medium', 0)} "
        f"low={by_severity.get('low', 0)}"
    )

    if not finding_rows:
        print("No findings. Dataset quality checks passed with clean outputs.")
        return 0

    print("Top Finding IDs:")
    for finding_id, count in id_counts.most_common(5):
        print(f"- {finding_id}: {count}")

    critical_or_high = [f for f in finding_rows if str(f.get("severity", "")).lower() in {"critical", "high"}]
    medium_rows = [f for f in finding_rows if str(f.get("severity", "")).lower() == "medium"]

    print("Action Priority:")
    if critical_or_high:
        top_ids = Counter(str(f.get("id", "UNKNOWN")) for f in critical_or_high).most_common(3)
        for finding_id, _count in top_ids:
            print(f"- {finding_id}: {_recommendation_for_id(finding_id)}")
    elif medium_rows:
        top_ids = Counter(str(f.get("id", "UNKNOWN")) for f in medium_rows).most_common(3)
        for finding_id, _count in top_ids:
            print(f"- {finding_id}: {_recommendation_for_id(finding_id)}")
    else:
        top_ids = Counter(str(f.get("id", "UNKNOWN")) for f in finding_rows).most_common(3)
        if set(fid for fid, _ in top_ids) == {"CLASS_LOW_SUPPORT"}:
            print("- Low-severity only: increase samples for under-supported classes to improve robustness.")
        else:
            for finding_id, _count in top_ids:
                print(f"- {finding_id}: {_recommendation_for_id(finding_id)}")

    return 0




def _extract_summary_counts(summary: dict, findings: list[dict]) -> tuple[dict[str, int], int]:
    totals = summary.get("totals", {})
    by_severity = totals.get("by_severity")
    findings_count = totals.get("findings")

    if not isinstance(by_severity, dict):
        by_severity = _counts_from_payload(findings)

    normalized = {
        "critical": int(by_severity.get("critical", 0)),
        "high": int(by_severity.get("high", 0)),
        "medium": int(by_severity.get("medium", 0)),
        "low": int(by_severity.get("low", 0)),
    }

    if not isinstance(findings_count, int):
        findings_count = len(findings)

    return normalized, findings_count


def _parse_flags_findings(flags: dict) -> list[dict]:
    findings = flags.get("findings", [])
    if not isinstance(findings, list):
        raise ExplainError("flags.json must contain a 'findings' array.")
    return [row for row in findings if isinstance(row, dict)]


def run_diff(args: argparse.Namespace) -> int:
    old_summary = _load_json(args.old / "summary.json", "summary.json")
    old_flags = _load_json(args.old / "flags.json", "flags.json")
    new_summary = _load_json(args.new / "summary.json", "summary.json")
    new_flags = _load_json(args.new / "flags.json", "flags.json")

    old_findings = _parse_flags_findings(old_flags)
    new_findings = _parse_flags_findings(new_flags)

    old_by_sev, old_total = _extract_summary_counts(old_summary, old_findings)
    new_by_sev, new_total = _extract_summary_counts(new_summary, new_findings)

    print("DQA Diff")
    print(f"old={args.old.as_posix()}")
    print(f"new={args.new.as_posix()}")
    print(f"findings: old={old_total} new={new_total} delta={new_total - old_total:+d}")

    print("severity_delta:")
    for sev in ["critical", "high", "medium", "low"]:
        delta = new_by_sev[sev] - old_by_sev[sev]
        print(f"- {sev}: old={old_by_sev[sev]} new={new_by_sev[sev]} delta={delta:+d}")

    old_id_counts = Counter(str(row.get("id", "UNKNOWN")) for row in old_findings)
    new_id_counts = Counter(str(row.get("id", "UNKNOWN")) for row in new_findings)

    all_ids = sorted(set(old_id_counts.keys()) | set(new_id_counts.keys()))
    deltas = {finding_id: new_id_counts.get(finding_id, 0) - old_id_counts.get(finding_id, 0) for finding_id in all_ids}

    regressions = [(finding_id, delta) for finding_id, delta in deltas.items() if delta > 0]
    improvements = [(finding_id, delta) for finding_id, delta in deltas.items() if delta < 0]

    regressions.sort(key=lambda row: row[1], reverse=True)
    improvements.sort(key=lambda row: row[1])

    if regressions:
        print("Top Regressions:")
        for finding_id, delta in regressions[:5]:
            print(f"- {finding_id}: +{delta}")
    else:
        print("Top Regressions:")
        print("- none")

    if improvements:
        print("Top Improvements:")
        for finding_id, delta in improvements[:5]:
            print(f"- {finding_id}: {delta}")
    else:
        print("Top Improvements:")
        print("- none")

    if args.fail_on_regression is None:
        return 0

    threshold = _SEVERITY_RANK[args.fail_on_regression]
    regress_fail = any(
        (new_by_sev[sev] - old_by_sev[sev]) > 0 and _SEVERITY_RANK[sev] >= threshold
        for sev in ["critical", "high", "medium", "low"]
    )

    if regress_fail:
        print(f"regression_gate=failed threshold={args.fail_on_regression}", file=sys.stderr)
        return 1

    print(f"regression_gate=passed threshold={args.fail_on_regression}")
    return 0

def run_audit(args: argparse.Namespace) -> int:
    start_perf = time.perf_counter()
    started_at = _utc_now()

    cfg = load_config(args.config)
    fail_on = args.fail_on or cfg.fail_on

    args.out.mkdir(parents=True, exist_ok=True)

    if args.remote_cache_ttl_hours is not None and args.remote_cache_ttl_hours < 0:
        raise RemoteDataError("--remote-cache-ttl-hours must be >= 0")

    data_yaml_path = resolve_data_yaml_source(
        args.data,
        args.data_url,
        args.out,
        data_url_format=args.data_url_format,
        roboflow_api_key=args.roboflow_api_key,
        use_remote_cache=not args.no_remote_cache,
        remote_cache_ttl_hours=args.remote_cache_ttl_hours,
    )

    requested_splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    # Auto-detect YOLO-vs-COCO from source path.
    source_suffix = data_yaml_path.suffix.lower()
    if data_yaml_path.is_file() and source_suffix in {".yaml", ".yml"}:
        spec = load_dataset_spec(data_yaml_path, requested_splits=requested_splits)
        index_result = build_index(spec, max_images=max(0, int(args.max_images)))
        dataset_ref = spec.data_yaml.as_posix()
        dataset_root = spec.root.as_posix()
        class_names = spec.names
        split_names = list(spec.splits.keys())
    elif data_yaml_path.is_dir() or (data_yaml_path.is_file() and source_suffix == ".json"):
        index_result = build_index_from_coco(data_yaml_path, requested_splits=requested_splits, max_images=max(0, int(args.max_images)))
        dataset_ref = str(index_result.payload.get("data_source", data_yaml_path.as_posix()))
        dataset_root = str(index_result.payload.get("dataset_root", ""))
        class_names = list(index_result.payload.get("class_names", []))
        split_names = sorted(index_result.payload.get("splits", {}).keys())
    else:
        raise RemoteDataError(f"Unsupported dataset source: {data_yaml_path}")

    write_json(args.out / "index.json", index_result.payload)

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
        integrity_findings = run_integrity(index_result.payload, class_count=index_result.class_count)
        findings.extend(integrity_findings)
        checks_summary["integrity"] = {"status": "completed", "counts": _counts(integrity_findings)}

    if cfg.checks.class_distribution.enabled:
        dist_findings = run_class_distribution(
            index_result.payload,
            class_count=index_result.class_count,
            min_instances_per_class_warn=cfg.checks.class_distribution.min_instances_per_class_warn,
            max_class_share_warn=cfg.checks.class_distribution.max_class_share_warn,
            split_drift_jsd_warn=cfg.checks.class_distribution.split_drift_jsd_warn,
            split_drift_jsd_high=cfg.checks.class_distribution.split_drift_jsd_high,
        )
        findings.extend(dist_findings)
        checks_summary["class_distribution"] = {"status": "completed", "counts": _counts(dist_findings)}

    if cfg.checks.bbox_sanity.enabled:
        bbox_findings = run_bbox_sanity(
            index_result.payload,
            min_box_area_ratio_warn=cfg.checks.bbox_sanity.min_box_area_ratio_warn,
            max_box_area_ratio_warn=cfg.checks.bbox_sanity.max_box_area_ratio_warn,
            max_boxes_per_image_warn=cfg.checks.bbox_sanity.max_boxes_per_image_warn,
            aspect_ratio_warn=cfg.checks.bbox_sanity.aspect_ratio_warn,
        )
        findings.extend(bbox_findings)
        checks_summary["bbox_sanity"] = {"status": "completed", "counts": _counts(bbox_findings)}

    if cfg.checks.duplicates.enabled:
        duplicate_findings = run_exact_duplicates(index_result.payload)
        findings.extend(duplicate_findings)
        checks_summary["duplicates"] = {"status": "completed", "counts": _counts(duplicate_findings)}

    near_dup_enabled = bool(cfg.checks.near_duplicates.enabled or args.near_dup)
    if near_dup_enabled:
        near_dup_findings, near_dup_reason = run_near_duplicates(
            index_result.payload,
            phash_hamming_threshold=cfg.checks.near_duplicates.phash_hamming_threshold,
        )
        findings.extend(near_dup_findings)
        status = "completed" if near_dup_reason is None else "skipped"
        checks_summary["near_duplicates"] = {"status": status, "counts": _counts(near_dup_findings)}

    if cfg.checks.leakage.enabled:
        leakage_findings = run_leakage(index_result.payload)
        findings.extend(leakage_findings)
        checks_summary["leakage"] = {"status": "completed", "counts": _counts(leakage_findings)}

    findings.sort(key=lambda f: (f.id, f.split or "", f.image or "", f.label or "", f.fingerprint))

    flags_payload = {
        "schema_version": "1.0.0",
        "findings": [_serialize_finding(f) for f in findings],
    }
    write_json(args.out / "flags.json", flags_payload)

    finished_at = _utc_now()
    duration_sec = round(time.perf_counter() - start_perf, 3)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    total_counts = _counts(findings)
    summary_payload = {
        "schema_version": "1.0.0",
        "run": {
            "run_id": run_id,
            "dqa_version": "0.1.0",
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_sec": duration_sec,
            "config": {"fail_on": fail_on, "enabled_checks": _enabled_checks(cfg, include_near_dup_flag=args.near_dup)},
        },
        "dataset": {
            "data_yaml": dataset_ref,
            "root": dataset_root,
            "splits": {
                split: {
                    "images": sum(1 for row in index_result.payload["images"] if row["split"] == split),
                    "labeled": sum(1 for row in index_result.payload["images"] if row["split"] == split and row["label_exists"]),
                    "unlabeled": sum(1 for row in index_result.payload["images"] if row["split"] == split and not row["label_exists"]),
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
            "build_failed": _fails_threshold(findings, fail_on),
        },
    }
    write_json(args.out / "summary.json", summary_payload)

    formats = {part.strip().lower() for part in str(args.format).split(",") if part.strip()}
    if "html" in formats:
        write_html(args.out / "report.html", summary_payload, flags_payload)

    run_log = [
        f"started_at={started_at}",
        f"finished_at={finished_at}",
        f"duration_sec={duration_sec}",
        f"data={dataset_ref}",
        f"out={args.out.resolve().as_posix()}",
        f"findings={len(findings)}",
        f"build_failed={summary_payload['totals']['build_failed']}",
    ]
    (args.out / "run.log").write_text("\n".join(run_log) + "\n", encoding="utf-8")

    print(f"findings={len(findings)}")
    print(f"build_failed={summary_payload['totals']['build_failed']}")

    return 1 if summary_payload["totals"]["build_failed"] else 0



def run_validate(args: argparse.Namespace) -> int:
    artifact = args.artifact
    schema = args.schema

    artifact_payload = _load_json(artifact, "artifact JSON")
    schema_payload = _load_json(schema, "schema JSON")

    try:
        Draft202012Validator.check_schema(schema_payload)
        validator = Draft202012Validator(schema_payload)
    except JsonSchemaError as exc:
        raise ValidateError(f"Invalid JSON schema: {schema}") from exc

    errors = sorted(validator.iter_errors(artifact_payload), key=lambda e: list(e.absolute_path))

    if not errors:
        print(f"valid artifact={artifact.as_posix()} schema={schema.as_posix()}")
        return 0

    print(f"invalid artifact={artifact.as_posix()} schema={schema.as_posix()}", file=sys.stderr)
    for err in errors[:10]:
        path = ".".join(str(part) for part in err.absolute_path) or "$"
        print(f"- {path}: {err.message}", file=sys.stderr)
    if len(errors) > 10:
        print(f"- ... and {len(errors) - 10} more validation error(s)", file=sys.stderr)
    return 1

def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "audit":
            return run_audit(args)
        if args.command == "explain":
            return run_explain(args)
        if args.command == "validate":
            return run_validate(args)
        if args.command == "diff":
            return run_diff(args)
        return 2
    except (ConfigError, DatasetSpecError, RemoteDataError, ExplainError, ValidateError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except JsonValidationError as exc:
        path = ".".join(str(part) for part in exc.absolute_path) or "$"
        print(f"validation error: {path}: {exc.message}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"runtime error: {exc}", file=sys.stderr)
        return 3






