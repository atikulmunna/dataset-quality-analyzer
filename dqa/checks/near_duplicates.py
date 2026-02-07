from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ..models import Finding

try:
    from PIL import Image
except ModuleNotFoundError:  # pragma: no cover
    Image = None


def _fp(*parts: str) -> str:
    raw = "|".join(parts).encode("utf-8")
    return "sha1:" + hashlib.sha1(raw).hexdigest()


def _ahash(image_path: Path, size: int = 8) -> int:
    with Image.open(image_path) as img:  # type: ignore[union-attr]
        gray = img.convert("L").resize((size, size))
        pixels = list(gray.getdata())
    avg = sum(pixels) / float(len(pixels))
    bits = 0
    for px in pixels:
        bits = (bits << 1) | (1 if px >= avg else 0)
    return bits


def _hamming_distance(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def run_near_duplicates(
    index_payload: dict[str, Any],
    phash_hamming_threshold: int,
) -> tuple[list[Finding], str | None]:
    if Image is None:
        return [], "Pillow not installed"

    dataset_root = Path(str(index_payload.get("dataset_root", ".")))

    rows = index_payload.get("images", [])
    hashes: list[tuple[dict[str, Any], int]] = []
    for row in rows:
        image_rel = str(row.get("image", ""))
        image_abs = dataset_root / image_rel
        try:
            h = _ahash(image_abs)
        except OSError:
            continue
        hashes.append((row, h))

    findings: list[Finding] = []
    n = len(hashes)
    for i in range(n):
        row_i, h_i = hashes[i]
        for j in range(i + 1, n):
            row_j, h_j = hashes[j]
            if row_i.get("sha256") == row_j.get("sha256"):
                continue
            dist = _hamming_distance(h_i, h_j)
            if dist > phash_hamming_threshold:
                continue

            split_i = str(row_i.get("split", ""))
            split_j = str(row_j.get("split", ""))
            image_i = str(row_i.get("image", ""))
            image_j = str(row_j.get("image", ""))

            across = split_i != split_j
            finding_id = "NEAR_DUPLICATE_ACROSS_SPLITS" if across else "NEAR_DUPLICATE_WITHIN_SPLIT"
            severity = "high" if across else "low"

            findings.append(
                Finding(
                    id=finding_id,
                    severity=severity,
                    split=split_i,
                    image=image_i,
                    message="Near-duplicate pair detected.",
                    metrics={
                        "pair_image": image_j,
                        "pair_split": split_j,
                        "hamming_distance": dist,
                        "threshold": phash_hamming_threshold,
                    },
                    fingerprint=_fp(finding_id, image_i, image_j, str(dist)),
                )
            )

    findings.sort(key=lambda f: (f.id, f.split or "", f.image or "", f.fingerprint))
    return findings, None
