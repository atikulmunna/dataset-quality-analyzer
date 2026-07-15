"""Finite AWS Batch worker orchestration for one hosted audit attempt."""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from typing import Any
from uuid import uuid4

from dqa.audit import AuditOptions, audit_dataset
from dqa.ingest import extract_validated_zip
from dqa.web.lifecycle import JobLifecycle, artifact_prefix

from .adapters import DynamoJobStore
from .observability import emit_event


_PRESETS = {
    "detection": "dqa.yaml",
    "segmentation": "dqa_seg.yaml",
    "segmentation_low_noise": "dqa_seg_low_noise.yaml",
}


class LeaseHeartbeat(AbstractContextManager["LeaseHeartbeat"]):
    def __init__(self, lifecycle: JobLifecycle, job_id: str, worker_id: str, *, interval_seconds: int = 60) -> None:
        self._lifecycle = lifecycle
        self._job_id = job_id
        self._worker_id = worker_id
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._failure: Exception | None = None
        self._thread = threading.Thread(target=self._run, name="dqa-heartbeat", daemon=True)

    def __enter__(self) -> "LeaseHeartbeat":
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._stop.set()
        self._thread.join(timeout=min(5, self._interval))

    def raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RuntimeError("Worker lost its lifecycle lease.") from self._failure

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            try:
                self._lifecycle.heartbeat(self._job_id, self._worker_id)
            except Exception as exc:
                self._failure = exc
                self._stop.set()


def discover_dataset(root: Path) -> Path:
    yaml_candidates = sorted(
        path for path in root.rglob("*") if path.is_file() and path.name.lower() in {"data.yaml", "data.yml"}
    )
    if len(yaml_candidates) == 1:
        return yaml_candidates[0]
    if len(yaml_candidates) > 1:
        raise ValueError("Archive contains multiple data.yaml entry points.")
    coco_candidates = sorted(
        path for path in root.rglob("*.json") if path.is_file() and "coco" in path.name.lower()
    )
    if not coco_candidates:
        raise ValueError("Archive contains neither one data.yaml nor COCO annotation files.")
    return root


def _error_code(exc: Exception) -> str:
    name = re.sub(r"(?<!^)(?=[A-Z])", "_", type(exc).__name__).lower()
    return f"worker_{name}"[:64]


def run_job(
    job_id: str,
    *,
    table: Any,
    s3: Any,
    bucket: str,
    workspace_root: Path,
    worker_id: str,
    config_root: Path,
) -> int:
    started_at = time.monotonic()
    emit_event("worker.started", job_id=job_id, worker_id=worker_id)
    store = DynamoJobStore(table)
    lifecycle = JobLifecycle(store)
    claimed = lifecycle.claim(job_id, worker_id)
    if claimed is None:
        current = store.get(job_id)
        if current is None or current.status != "running" or current.lease_until is None:
            emit_event("worker.skipped", job_id=job_id, worker_id=worker_id, reason="not_claimable")
            return 0
        lease_until = datetime.fromisoformat(current.lease_until.replace("Z", "+00:00"))
        wait_seconds = max(0.0, (lease_until - datetime.now(timezone.utc)).total_seconds())
        time.sleep(min(wait_seconds + 1.0, 301.0))
        claimed = lifecycle.claim(job_id, worker_id)
        if claimed is None:
            emit_event("worker.skipped", job_id=job_id, worker_id=worker_id, reason="lease_not_reclaimed")
            return 0

    emit_event("worker.claimed", job_id=job_id, worker_id=worker_id, attempt=claimed.attempt)
    try:
        with tempfile.TemporaryDirectory(prefix=f"dqa-{job_id[:12]}-", dir=workspace_root) as raw:
            workspace = Path(raw)
            archive = workspace / "dataset.zip"
            extracted = workspace / "dataset"
            output = workspace / "out"
            s3.download_file(bucket, claimed.dataset_key, str(archive))
            extract_validated_zip(archive, extracted)
            source = discover_dataset(extracted)
            preset_path = config_root / _PRESETS[claimed.preset]
            with LeaseHeartbeat(lifecycle, job_id, worker_id) as heartbeat:
                audit_dataset(
                    AuditOptions(
                        data=source,
                        out=output,
                        config=preset_path,
                        workers=min(4, os.cpu_count() or 1),
                        max_images=25_000,
                        near_duplicates=claimed.near_duplicates,
                        formats=("html", "json"),
                        fail_on=claimed.fail_on,
                    )
                )
                heartbeat.raise_if_failed()
            prefix = artifact_prefix(claimed)
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    key = prefix + path.relative_to(output).as_posix()
                    s3.upload_file(str(path), bucket, key)
            lifecycle.complete(job_id, worker_id, prefix)
        emit_event(
            "worker.succeeded",
            job_id=job_id,
            worker_id=worker_id,
            attempt=claimed.attempt,
            duration_seconds=round(time.monotonic() - started_at, 3),
        )
        return 0
    except Exception as exc:
        error_code = _error_code(exc)
        emit_event(
            "worker.failed",
            job_id=job_id,
            worker_id=worker_id,
            attempt=claimed.attempt,
            duration_seconds=round(time.monotonic() - started_at, 3),
            error_code=error_code,
        )
        lifecycle.fail(job_id, worker_id, error_code)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one hosted DQA Batch job")
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args(argv)

    import boto3

    table_name = os.environ.get("DQA_TABLE_NAME")
    bucket = os.environ.get("DQA_DATA_BUCKET")
    if not table_name or not bucket:
        raise RuntimeError("DQA_TABLE_NAME and DQA_DATA_BUCKET are required.")
    batch_job_id = os.environ.get("AWS_BATCH_JOB_ID", "local")
    worker_id = f"{batch_job_id}:{uuid4().hex[:12]}"
    workspace = Path(os.environ.get("DQA_WORKSPACE", "/workspace"))
    workspace.mkdir(parents=True, exist_ok=True)
    return run_job(
        args.job_id,
        table=boto3.resource("dynamodb").Table(table_name),
        s3=boto3.client("s3"),
        bucket=bucket,
        workspace_root=workspace,
        worker_id=worker_id,
        config_root=Path(os.environ.get("DQA_CONFIG_ROOT", "/opt/dqa/configs")),
    )


if __name__ == "__main__":
    raise SystemExit(main())
