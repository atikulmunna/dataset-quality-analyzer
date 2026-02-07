from __future__ import annotations

import hashlib
from typing import Any

from ..models import Finding


def _fp(*parts: str) -> str:
    raw = "|".join(parts).encode("utf-8")
    return "sha1:" + hashlib.sha1(raw).hexdigest()


def run_integrity(index_payload: dict[str, Any], class_count: int) -> list[Finding]:
    findings: list[Finding] = []

    for row in index_payload.get("images", []):
        split = str(row.get("split", ""))
        image = str(row.get("image", ""))
        label = row.get("label")

        if not row.get("label_exists", False):
            findings.append(
                Finding(
                    id="INTEGRITY_MISSING_LABEL",
                    severity="high",
                    message="Image has no matching label file.",
                    split=split,
                    image=image,
                    label=None,
                    fingerprint=_fp("INTEGRITY_MISSING_LABEL", split, image),
                )
            )

        image_error = row.get("image_error")
        if image_error:
            findings.append(
                Finding(
                    id="INTEGRITY_CORRUPT_IMAGE",
                    severity="critical",
                    message=f"Image could not be decoded: {image_error}",
                    split=split,
                    image=image,
                    label=label,
                    fingerprint=_fp("INTEGRITY_CORRUPT_IMAGE", split, image),
                )
            )

        for err in row.get("label_parse_errors", []):
            line = int(err.get("line", 0))
            reason = str(err.get("reason", "parse_error"))
            findings.append(
                Finding(
                    id="INTEGRITY_MALFORMED_ROW",
                    severity="high",
                    message=f"Malformed label row at line {line}: {reason}",
                    split=split,
                    image=image,
                    label=label,
                    metrics={"line": line, "reason": reason},
                    fingerprint=_fp("INTEGRITY_MALFORMED_ROW", split, image, str(line), reason),
                )
            )

        for parsed in row.get("label_rows", []):
            class_id = int(parsed.get("class_id", -1))
            line = int(parsed.get("line", 0))
            annotation_type = str(parsed.get("annotation_type", "bbox"))

            if class_id < 0 or class_id >= class_count:
                findings.append(
                    Finding(
                        id="INTEGRITY_INVALID_CLASS_ID",
                        severity="high",
                        message=f"Class ID {class_id} is outside [0, {max(class_count - 1, 0)}].",
                        split=split,
                        image=image,
                        label=label,
                        class_id=class_id,
                        metrics={"line": line},
                        fingerprint=_fp("INTEGRITY_INVALID_CLASS_ID", split, image, str(line), str(class_id)),
                    )
                )

            if annotation_type == "segment":
                coords = parsed.get("coords", [])
                if not isinstance(coords, list):
                    coords = []
                out_of_range = any(float(v) < 0.0 or float(v) > 1.0 for v in coords)
                if out_of_range:
                    findings.append(
                        Finding(
                            id="INTEGRITY_COORD_OUT_OF_RANGE",
                            severity="high",
                            message="Polygon values must be normalized to [0,1].",
                            split=split,
                            image=image,
                            label=label,
                            class_id=class_id if class_id >= 0 else None,
                            metrics={"line": line, "annotation_type": "segment"},
                            fingerprint=_fp("INTEGRITY_COORD_OUT_OF_RANGE", split, image, str(line)),
                        )
                    )
                continue

            x_center = float(parsed.get("x_center", 0.0))
            y_center = float(parsed.get("y_center", 0.0))
            width = float(parsed.get("width", 0.0))
            height = float(parsed.get("height", 0.0))
            coords = [x_center, y_center, width, height]
            if any(v < 0.0 or v > 1.0 for v in coords):
                findings.append(
                    Finding(
                        id="INTEGRITY_COORD_OUT_OF_RANGE",
                        severity="high",
                        message="BBox values must be normalized to [0,1].",
                        split=split,
                        image=image,
                        label=label,
                        class_id=class_id if class_id >= 0 else None,
                        metrics={
                            "line": line,
                            "annotation_type": "bbox",
                            "x_center": x_center,
                            "y_center": y_center,
                            "width": width,
                            "height": height,
                        },
                        fingerprint=_fp("INTEGRITY_COORD_OUT_OF_RANGE", split, image, str(line)),
                    )
                )

    for split_name, split_meta in index_payload.get("splits", {}).items():
        for orphan_label in split_meta.get("orphan_labels", []):
            findings.append(
                Finding(
                    id="INTEGRITY_ORPHAN_LABEL",
                    severity="medium",
                    message="Label file has no matching image file.",
                    split=str(split_name),
                    label=str(orphan_label),
                    fingerprint=_fp("INTEGRITY_ORPHAN_LABEL", str(split_name), str(orphan_label)),
                )
            )

    findings.sort(key=lambda f: (f.id, f.split or "", f.image or "", f.label or "", f.fingerprint))
    return findings
