from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Any

from ..models import Finding


def _fp(*parts: str) -> str:
    raw = "|".join(parts).encode("utf-8")
    return "sha1:" + hashlib.sha1(raw).hexdigest()


def _distribution(counts: Counter[int], class_count: int) -> list[float]:
    total = float(sum(counts.values()))
    if total <= 0:
        return [0.0 for _ in range(class_count)]
    return [counts.get(i, 0) / total for i in range(class_count)]


def _jsd(p: list[float], q: list[float]) -> float:
    def _kl(a: list[float], b: list[float]) -> float:
        s = 0.0
        for ai, bi in zip(a, b):
            if ai > 0 and bi > 0:
                s += ai * math.log2(ai / bi)
        return s

    m = [(pi + qi) / 2.0 for pi, qi in zip(p, q)]
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def run_class_distribution(
    index_payload: dict[str, Any],
    class_count: int,
    min_instances_per_class_warn: int,
    max_class_share_warn: float,
    split_drift_jsd_warn: float,
    split_drift_jsd_high: float,
) -> list[Finding]:
    findings: list[Finding] = []

    global_counts: Counter[int] = Counter()
    split_counts: dict[str, Counter[int]] = defaultdict(Counter)

    for row in index_payload.get("images", []):
        split = str(row.get("split", ""))
        for parsed in row.get("label_rows", []):
            class_id = int(parsed.get("class_id", -1))
            if 0 <= class_id < class_count:
                global_counts[class_id] += 1
                split_counts[split][class_id] += 1

    total_instances = sum(global_counts.values())
    if total_instances > 0:
        dominant_class, dominant_count = max(global_counts.items(), key=lambda x: x[1])
        dominant_share = dominant_count / float(total_instances)
        if dominant_share > max_class_share_warn:
            findings.append(
                Finding(
                    id="CLASS_IMBALANCE_HIGH",
                    severity="medium",
                    message="Dominant class share exceeds configured threshold.",
                    class_id=int(dominant_class),
                    metrics={
                        "dominant_class": int(dominant_class),
                        "dominant_share": dominant_share,
                        "threshold": max_class_share_warn,
                    },
                    fingerprint=_fp("CLASS_IMBALANCE_HIGH", str(dominant_class), f"{dominant_share:.6f}"),
                )
            )

    for class_id in range(class_count):
        count = int(global_counts.get(class_id, 0))
        if count < min_instances_per_class_warn:
            findings.append(
                Finding(
                    id="CLASS_LOW_SUPPORT",
                    severity="low",
                    message="Class support is below configured minimum.",
                    class_id=class_id,
                    metrics={"count": count, "threshold": min_instances_per_class_warn},
                    fingerprint=_fp("CLASS_LOW_SUPPORT", str(class_id), str(count)),
                )
            )

    train_dist = _distribution(split_counts.get("train", Counter()), class_count)
    for target in ["val", "test"]:
        target_dist = _distribution(split_counts.get(target, Counter()), class_count)
        if not any(train_dist) or not any(target_dist):
            continue
        jsd = _jsd(train_dist, target_dist)
        if jsd >= split_drift_jsd_warn:
            findings.append(
                Finding(
                    id="CLASS_SPLIT_DRIFT",
                    severity="medium",
                    split=target,
                    message="Class distribution drift exceeds configured JSD threshold.",
                    metrics={
                        "jsd": jsd,
                        "warn_threshold": split_drift_jsd_warn,
                        "high_threshold": split_drift_jsd_high,
                        "reference_split": "train",
                        "target_split": target,
                    },
                    fingerprint=_fp("CLASS_SPLIT_DRIFT", target, f"{jsd:.6f}"),
                )
            )

    findings.sort(key=lambda f: (f.id, f.split or "", str(f.class_id) if f.class_id is not None else "", f.fingerprint))
    return findings
