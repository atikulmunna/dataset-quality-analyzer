from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from ..models import Finding


def _fp(*parts: str) -> str:
    raw = "|".join(parts).encode("utf-8")
    return "sha1:" + hashlib.sha1(raw).hexdigest()


def run_leakage(index_payload: dict[str, Any]) -> list[Finding]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in index_payload.get("images", []):
        sha = str(row.get("sha256", ""))
        if sha:
            by_hash[sha].append(row)

    findings: list[Finding] = []

    for sha, rows in by_hash.items():
        train_rows = [r for r in rows if r.get("split") == "train"]
        val_rows = [r for r in rows if r.get("split") == "val"]
        test_rows = [r for r in rows if r.get("split") == "test"]

        if train_rows and val_rows:
            for row in val_rows:
                split = str(row.get("split", ""))
                image = str(row.get("image", ""))
                findings.append(
                    Finding(
                        id="LEAKAGE_EXACT_TRAIN_VAL",
                        severity="critical",
                        split=split,
                        image=image,
                        label=row.get("label"),
                        message="Exact train/val leakage detected.",
                        metrics={
                            "sha256": sha,
                            "matching_train_images": [str(r.get("image", "")) for r in train_rows],
                        },
                        fingerprint=_fp("LEAKAGE_EXACT_TRAIN_VAL", sha, split, image),
                    )
                )

        if train_rows and test_rows:
            for row in test_rows:
                split = str(row.get("split", ""))
                image = str(row.get("image", ""))
                findings.append(
                    Finding(
                        id="LEAKAGE_EXACT_TRAIN_TEST",
                        severity="critical",
                        split=split,
                        image=image,
                        label=row.get("label"),
                        message="Exact train/test leakage detected.",
                        metrics={
                            "sha256": sha,
                            "matching_train_images": [str(r.get("image", "")) for r in train_rows],
                        },
                        fingerprint=_fp("LEAKAGE_EXACT_TRAIN_TEST", sha, split, image),
                    )
                )

    findings.sort(key=lambda f: (f.id, f.split or "", f.image or "", f.fingerprint))
    return findings
