from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError as JsonSchemaError
from jsonschema.exceptions import ValidationError as JsonValidationError

from .audit import AuditOptions, audit_dataset
from .config import ConfigError
from .io_yolo import DatasetSpecError
from .remote import RemoteDataError


_SEVERITY_RANK = {"critical": 4, "high": 3, "medium": 2, "low": 1}


class ExplainError(ValueError):
    """Raised when explain input artifacts are invalid."""


class ValidateError(ValueError):
    """Raised when validate inputs are invalid."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dqa", description="Dataset Quality Analyzer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="Audit a YOLO or COCO dataset")
    audit.add_argument("--data", type=Path, help="Path to local data.yaml")
    audit.add_argument("--data-url", type=str, help="Remote dataset URL (currently supports Roboflow URLs)")
    audit.add_argument("--data-url-format", type=str, default="yolov11", help="Remote export format (default: yolov11)")
    audit.add_argument("--roboflow-api-key", type=str, default=None, help="Optional Roboflow API key override")
    audit.add_argument("--no-remote-cache", action="store_true", help="Disable remote dataset cache reuse")
    audit.add_argument("--remote-cache-ttl-hours", type=float, default=None, help="Remote cache TTL in hours (default: no TTL)")
    audit.add_argument("--out", required=True, type=Path, help="Output directory")
    audit.add_argument("--config", type=Path, default=None, help="Config path (default: built-in detection settings)")
    audit.add_argument("--splits", default="train,val,test", help="Comma-separated splits")
    audit.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1), help="Hash workers (1-32; default: up to 4)")
    audit.add_argument("--max-images", type=int, default=0)
    audit.add_argument("--near-dup", action="store_true")
    audit.add_argument("--format", default="html,json")
    audit.add_argument("--fail-on", choices=["critical", "high", "medium", "low"], default=None)

    explain = subparsers.add_parser("explain", help="Explain findings from a completed DQA run")
    explain.add_argument("--run", type=Path, default=None, help="Run directory containing summary.json and flags.json")
    explain.add_argument("--summary", type=Path, default=None, help="Path to summary.json")
    explain.add_argument("--flags", type=Path, default=None, help="Path to flags.json")
    explain.add_argument("--format", choices=["text", "markdown", "json"], default="text", help="Explain output format")
    explain.add_argument("--out-file", type=Path, default=None, help="Optional output file for explain content")

    validate = subparsers.add_parser("validate", help="Validate a DQA JSON artifact against a JSON schema")
    validate.add_argument("--artifact", required=True, type=Path, help="Path to artifact JSON (e.g., summary.json)")
    validate.add_argument("--schema", required=True, type=Path, help="Path to JSON schema")

    diff = subparsers.add_parser("diff", help="Compare two DQA runs and show finding regressions")
    diff.add_argument("--old", required=True, type=Path, help="Old run directory containing summary.json and flags.json")
    diff.add_argument("--new", required=True, type=Path, help="New run directory containing summary.json and flags.json")
    diff.add_argument("--fail-on-regression", choices=["critical", "high", "medium", "low"], default=None)

    return parser


def _counts_from_payload(findings: list[dict]) -> dict[str, int]:
    counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for row in findings:
        sev = str(row.get("severity", "")).lower()
        if sev in counts:
            counts[sev] += 1
    return counts


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

    output_format = str(getattr(args, "format", "text")).lower()
    out_file = getattr(args, "out_file", None)
    if output_format not in {"text", "markdown", "json"}:
        raise ExplainError(f"Unsupported explain format: {output_format}")

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

    top_finding_ids = id_counts.most_common(5)
    critical_or_high = [f for f in finding_rows if str(f.get("severity", "")).lower() in {"critical", "high"}]
    medium_rows = [f for f in finding_rows if str(f.get("severity", "")).lower() == "medium"]

    action_priority: list[str] = []
    if finding_rows:
        if critical_or_high:
            top_ids = Counter(str(f.get("id", "UNKNOWN")) for f in critical_or_high).most_common(3)
            action_priority.extend([f"{finding_id}: {_recommendation_for_id(finding_id)}" for finding_id, _ in top_ids])
        elif medium_rows:
            top_ids = Counter(str(f.get("id", "UNKNOWN")) for f in medium_rows).most_common(3)
            action_priority.extend([f"{finding_id}: {_recommendation_for_id(finding_id)}" for finding_id, _ in top_ids])
        else:
            top_ids = Counter(str(f.get("id", "UNKNOWN")) for f in finding_rows).most_common(3)
            if set(fid for fid, _ in top_ids) == {"CLASS_LOW_SUPPORT"}:
                action_priority.append("Low-severity only: increase samples for under-supported classes to improve robustness.")
            else:
                action_priority.extend([f"{finding_id}: {_recommendation_for_id(finding_id)}" for finding_id, _ in top_ids])

    payload = {
        "title": "DQA Explain",
        "run": run_label,
        "findings": len(finding_rows),
        "build_failed": totals.get("build_failed"),
        "severity": {
            "critical": by_severity.get("critical", 0),
            "high": by_severity.get("high", 0),
            "medium": by_severity.get("medium", 0),
            "low": by_severity.get("low", 0),
        },
        "top_finding_ids": [{"id": finding_id, "count": count} for finding_id, count in top_finding_ids],
        "action_priority": action_priority,
    }

    if output_format == "json":
        output_text = json.dumps(payload, indent=2)
    elif output_format == "markdown":
        lines = [
            "# DQA Explain",
            f"- run: `{run_label}`",
            f"- findings: **{len(finding_rows)}**",
            f"- build_failed: **{totals.get('build_failed')}**",
            "",
            "## Severity",
            f"- critical: {by_severity.get('critical', 0)}",
            f"- high: {by_severity.get('high', 0)}",
            f"- medium: {by_severity.get('medium', 0)}",
            f"- low: {by_severity.get('low', 0)}",
            "",
        ]
        if finding_rows:
            lines.append("## Top Finding IDs")
            for finding_id, count in top_finding_ids:
                lines.append(f"- `{finding_id}`: {count}")
            lines.append("")
            lines.append("## Action Priority")
            for action in action_priority:
                lines.append(f"- {action}")
        else:
            lines.append("No findings. Dataset quality checks passed with clean outputs.")
        output_text = "\n".join(lines)
    else:
        lines = [
            "DQA Explain",
            f"run={run_label}",
            f"findings={len(finding_rows)} build_failed={totals.get('build_failed')}",
            (
                "severity: "
                f"critical={by_severity.get('critical', 0)} "
                f"high={by_severity.get('high', 0)} "
                f"medium={by_severity.get('medium', 0)} "
                f"low={by_severity.get('low', 0)}"
            ),
        ]
        if not finding_rows:
            lines.append("No findings. Dataset quality checks passed with clean outputs.")
        else:
            lines.append("Top Finding IDs:")
            for finding_id, count in top_finding_ids:
                lines.append(f"- {finding_id}: {count}")
            lines.append("Action Priority:")
            for action in action_priority:
                lines.append(f"- {action}")
        output_text = "\n".join(lines)

    if out_file is not None:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        out_file.write_text(output_text + "\n", encoding="utf-8")
    else:
        print(output_text)

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
    splits = tuple(part.strip() for part in args.splits.split(",") if part.strip())
    formats = tuple(part.strip().lower() for part in str(args.format).split(",") if part.strip())
    result = audit_dataset(
        AuditOptions(
            data=args.data,
            data_url=args.data_url,
            data_url_format=args.data_url_format,
            roboflow_api_key=args.roboflow_api_key,
            use_remote_cache=not args.no_remote_cache,
            remote_cache_ttl_hours=args.remote_cache_ttl_hours,
            out=args.out,
            config=args.config,
            splits=splits,
            workers=args.workers,
            max_images=max(0, int(args.max_images)),
            near_duplicates=args.near_dup,
            formats=formats,
            fail_on=args.fail_on,
        )
    )
    totals = result.summary["totals"]
    print(f"findings={totals['findings']}")
    print(f"build_failed={totals['build_failed']}")
    return result.exit_code

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
