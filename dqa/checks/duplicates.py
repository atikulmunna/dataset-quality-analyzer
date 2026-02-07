from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from ..models import Finding


def _fp(*parts: str) -> str:
    raw = "|".join(parts).encode("utf-8")
    return "sha1:" + hashlib.sha1(raw).hexdigest()


def run_exact_duplicates(index_payload: dict[str, Any]) -> list[Finding]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in index_payload.get("images", []):
        sha = str(row.get("sha256", ""))
        if sha:
            by_hash[sha].append(row)

    findings: list[Finding] = []
    for sha, rows in by_hash.items():
        if len(rows) < 2:
            continue
        splits = sorted({str(r.get("split", "")) for r in rows})
        across = len(splits) > 1
        finding_id = "DUPLICATE_ACROSS_SPLITS" if across else "DUPLICATE_WITHIN_SPLIT"
        severity = "high" if across else "medium"

        for row in rows:
            split = str(row.get("split", ""))
            image = str(row.get("image", ""))
            findings.append(
                Finding(
                    id=finding_id,
                    severity=severity,
                    split=split,
                    image=image,
                    label=row.get("label"),
                    message=f"Exact duplicate cluster detected (size={len(rows)}).",
                    metrics={"sha256": sha, "cluster_size": len(rows), "splits": splits},
                    fingerprint=_fp(finding_id, sha, split, image),
                )
            )

    findings.sort(key=lambda f: (f.id, f.split or "", f.image or "", f.fingerprint))
    return findings
