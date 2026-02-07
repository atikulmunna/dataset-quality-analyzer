from __future__ import annotations

import hashlib
from typing import Any

from ..models import Finding


def _fp(*parts: str) -> str:
    raw = "|".join(parts).encode("utf-8")
    return "sha1:" + hashlib.sha1(raw).hexdigest()


def run_bbox_sanity(
    index_payload: dict[str, Any],
    min_box_area_ratio_warn: float,
    max_box_area_ratio_warn: float,
    max_boxes_per_image_warn: int,
    aspect_ratio_warn: float,
) -> list[Finding]:
    findings: list[Finding] = []

    for row in index_payload.get("images", []):
        split = str(row.get("split", ""))
        image = str(row.get("image", ""))
        label = row.get("label")
        label_rows = row.get("label_rows", [])

        if len(label_rows) > max_boxes_per_image_warn:
            findings.append(
                Finding(
                    id="BBOX_TOO_MANY_PER_IMAGE",
                    severity="medium",
                    split=split,
                    image=image,
                    label=label,
                    message="Image has more boxes than configured threshold.",
                    metrics={"count": len(label_rows), "threshold": max_boxes_per_image_warn},
                    fingerprint=_fp("BBOX_TOO_MANY_PER_IMAGE", split, image, str(len(label_rows))),
                )
            )

        for parsed in label_rows:
            class_id = int(parsed.get("class_id", -1))
            line = int(parsed.get("line", 0))
            width = float(parsed.get("width", 0.0))
            height = float(parsed.get("height", 0.0))

            area = width * height
            if area < min_box_area_ratio_warn:
                findings.append(
                    Finding(
                        id="BBOX_TINY_BOX",
                        severity="medium",
                        split=split,
                        image=image,
                        label=label,
                        class_id=class_id if class_id >= 0 else None,
                        message="Bounding box area is below configured threshold.",
                        metrics={"line": line, "area": area, "threshold": min_box_area_ratio_warn},
                        fingerprint=_fp("BBOX_TINY_BOX", split, image, str(line)),
                    )
                )

            if area > max_box_area_ratio_warn:
                findings.append(
                    Finding(
                        id="BBOX_OVERSIZED_BOX",
                        severity="medium",
                        split=split,
                        image=image,
                        label=label,
                        class_id=class_id if class_id >= 0 else None,
                        message="Bounding box area is above configured threshold.",
                        metrics={"line": line, "area": area, "threshold": max_box_area_ratio_warn},
                        fingerprint=_fp("BBOX_OVERSIZED_BOX", split, image, str(line)),
                    )
                )

            if width > 0 and height > 0:
                aspect = max(width / height, height / width)
                if aspect > aspect_ratio_warn:
                    findings.append(
                        Finding(
                            id="BBOX_EXTREME_ASPECT_RATIO",
                            severity="medium",
                            split=split,
                            image=image,
                            label=label,
                            class_id=class_id if class_id >= 0 else None,
                            message="Bounding box aspect ratio exceeds configured threshold.",
                            metrics={"line": line, "aspect_ratio": aspect, "threshold": aspect_ratio_warn},
                            fingerprint=_fp("BBOX_EXTREME_ASPECT_RATIO", split, image, str(line)),
                        )
                    )

    findings.sort(key=lambda f: (f.id, f.split or "", f.image or "", f.fingerprint))
    return findings
