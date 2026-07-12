from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
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


def _candidate_pairs(hashes: list[int], threshold: int) -> list[tuple[int, int]]:
    """Return every matching pair using an exact Hamming-distance BK-tree."""
    if threshold >= 64:
        return [(i, j) for i in range(len(hashes)) for j in range(i + 1, len(hashes))]
    if not hashes:
        return []

    nodes: list[dict[str, Any]] = [{"value": hashes[0], "indices": [0], "children": {}}]
    pairs: list[tuple[int, int]] = []

    for index, value in enumerate(hashes[1:], start=1):
        stack = [0]
        while stack:
            node_index = stack.pop()
            node = nodes[node_index]
            distance = _hamming_distance(value, int(node["value"]))
            if distance <= threshold:
                pairs.extend((other, index) for other in node["indices"])
            lower = max(0, distance - threshold)
            upper = min(64, distance + threshold)
            stack.extend(
                child
                for edge, child in node["children"].items()
                if lower <= edge <= upper
            )

        node_index = 0
        while True:
            node = nodes[node_index]
            distance = _hamming_distance(value, int(node["value"]))
            if distance == 0:
                node["indices"].append(index)
                break
            child = node["children"].get(distance)
            if child is None:
                node["children"][distance] = len(nodes)
                nodes.append({"value": value, "indices": [index], "children": {}})
                break
            node_index = child

    return pairs


def run_near_duplicates(
    index_payload: dict[str, Any],
    phash_hamming_threshold: int,
    workers: int = 1,
) -> tuple[list[Finding], str | None]:
    if Image is None:
        return [], "Pillow not installed"

    dataset_root = Path(str(index_payload.get("dataset_root", ".")))

    rows = index_payload.get("images", [])
    readable: list[tuple[dict[str, Any], Path]] = []
    for row in rows:
        image_rel = str(row.get("image", ""))
        image_abs = dataset_root / image_rel
        readable.append((row, image_abs))

    def hash_image(item: tuple[dict[str, Any], Path]) -> tuple[dict[str, Any], int] | None:
        row, image_abs = item
        try:
            h = _ahash(image_abs)
        except OSError:
            return None
        return row, h

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            hashed = list(executor.map(hash_image, readable))
    else:
        hashed = [hash_image(item) for item in readable]
    hashes = [item for item in hashed if item is not None]

    findings: list[Finding] = []
    candidates = _candidate_pairs([item[1] for item in hashes], phash_hamming_threshold)
    for i, j in candidates:
        row_i, h_i = hashes[i]
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
