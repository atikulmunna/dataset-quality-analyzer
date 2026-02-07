from __future__ import annotations

import hashlib
import json
import struct
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io_yolo import DatasetSpec

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class BuildIndexResult:
    payload: dict[str, Any]
    class_count: int


def _read_png_size(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("invalid png")
    width, height = struct.unpack(">II", data[16:24])
    return int(width), int(height)


def _read_jpeg_size(data: bytes) -> tuple[int, int]:
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        raise ValueError("invalid jpeg")
    idx = 2
    data_len = len(data)
    while idx + 9 < data_len:
        if data[idx] != 0xFF:
            idx += 1
            continue
        marker = data[idx + 1]
        idx += 2
        if marker in {0xD8, 0xD9}:
            continue
        if idx + 2 > data_len:
            break
        seg_len = struct.unpack(">H", data[idx:idx + 2])[0]
        if seg_len < 2 or idx + seg_len > data_len:
            break
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if idx + 7 > data_len:
                break
            height, width = struct.unpack(">HH", data[idx + 3:idx + 7])
            return int(width), int(height)
        idx += seg_len
    raise ValueError("jpeg size parse failed")


def _image_dimensions(path: Path) -> tuple[int | None, int | None, str | None]:
    try:
        with path.open("rb") as f:
            head = f.read(65536)
    except OSError as exc:
        return None, None, str(exc)

    try:
        if head.startswith(b"\x89PNG\r\n\x1a\n"):
            width, height = _read_png_size(head)
            return width, height, None
        if head.startswith(b"\xff\xd8"):
            width, height = _read_jpeg_size(head)
            return width, height, None
    except ValueError as exc:
        return None, None, str(exc)

    return None, None, None


def _sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
    return hasher.hexdigest()


def _parse_label_rows(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()

        if len(parts) == 5:
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
            except ValueError:
                errors.append({"line": line_no, "raw": raw_line, "reason": "non_numeric"})
                continue
            rows.append(
                {
                    "line": line_no,
                    "annotation_type": "bbox",
                    "annotation_source": "yolo",
                    "class_id": class_id,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                }
            )
            continue

        if len(parts) >= 7 and len(parts) % 2 == 1:
            try:
                class_id = int(parts[0])
                coords = [float(v) for v in parts[1:]]
            except ValueError:
                errors.append({"line": line_no, "raw": raw_line, "reason": "non_numeric"})
                continue

            if len(coords) < 6:
                errors.append({"line": line_no, "raw": raw_line, "reason": "min_polygon_points"})
                continue

            xs = coords[0::2]
            ys = coords[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            rows.append(
                {
                    "line": line_no,
                    "annotation_type": "segment",
                    "annotation_source": "yolo",
                    "class_id": class_id,
                    "x_center": (x_min + x_max) / 2.0,
                    "y_center": (y_min + y_max) / 2.0,
                    "width": x_max - x_min,
                    "height": y_max - y_min,
                    "coords": coords,
                }
            )
            continue

        errors.append({"line": line_no, "raw": raw_line, "reason": "expected_5_tokens_or_valid_polygon"})

    return rows, errors


def build_index(spec: DatasetSpec, max_images: int = 0) -> BuildIndexResult:
    images: list[dict[str, Any]] = []
    split_stats: dict[str, dict[str, Any]] = {}

    total_seen = 0
    for split_name in sorted(spec.splits):
        split = spec.splits[split_name]
        split_images: list[Path] = []
        if split.images_dir.exists() and split.images_dir.is_dir():
            split_images = sorted(
                p for p in split.images_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
            )

        label_set: set[Path] = set()
        if split.labels_dir.exists() and split.labels_dir.is_dir():
            label_set = {p.resolve() for p in split.labels_dir.rglob("*.txt") if p.is_file()}

        matched_labels: set[Path] = set()

        for image_path in split_images:
            if max_images > 0 and total_seen >= max_images:
                break
            total_seen += 1

            image_rel = image_path.relative_to(spec.root).as_posix()
            label_path = split.labels_dir / f"{image_path.stem}.txt"
            label_exists = label_path.exists() and label_path.is_file()

            if label_exists:
                matched_labels.add(label_path.resolve())

            try:
                stat = image_path.stat()
                size_bytes = int(stat.st_size)
                mtime_ns = int(stat.st_mtime_ns)
            except OSError:
                size_bytes = -1
                mtime_ns = -1

            width, height, image_error = _image_dimensions(image_path)

            label_rows: list[dict[str, Any]] = []
            parse_errors: list[dict[str, Any]] = []
            if label_exists:
                try:
                    label_rows, parse_errors = _parse_label_rows(label_path)
                except OSError as exc:
                    parse_errors = [{"line": 0, "raw": "", "reason": f"read_error:{exc}"}]

            images.append(
                {
                    "split": split_name,
                    "image": image_rel,
                    "label": label_path.relative_to(spec.root).as_posix() if label_exists else None,
                    "label_exists": label_exists,
                    "size_bytes": size_bytes,
                    "mtime_ns": mtime_ns,
                    "sha256": _sha256_file(image_path),
                    "width": width,
                    "height": height,
                    "image_error": image_error,
                    "label_rows": label_rows,
                    "label_parse_errors": parse_errors,
                }
            )

        orphan_labels = sorted(
            p.relative_to(spec.root).as_posix()
            for p in label_set
            if p not in matched_labels
        )

        split_stats[split_name] = {
            "images_dir": split.images_dir.as_posix(),
            "labels_dir": split.labels_dir.as_posix(),
            "image_count": len(split_images),
            "orphan_labels": orphan_labels,
        }

        if max_images > 0 and total_seen >= max_images:
            break

    images.sort(key=lambda x: (x["split"], x["image"]))

    cache_basis = []
    for row in images:
        cache_basis.append(f"{row['image']}|{row['size_bytes']}|{row['mtime_ns']}|{row['sha256']}")
    digest = hashlib.sha256("\n".join(cache_basis).encode("utf-8")).hexdigest()

    payload = {
        "schema_version": "1.0.0",
        "dataset_root": spec.root.as_posix(),
        "data_yaml": spec.data_yaml.as_posix(),
        "class_names": spec.names,
        "splits": split_stats,
        "images": images,
        "cache_key": f"sha256:{digest}",
    }
    return BuildIndexResult(payload=payload, class_count=len(spec.names))


def _infer_split(name: str) -> str | None:
    n = name.lower()
    if n in {"train", "training"}:
        return "train"
    if n in {"val", "valid", "validation"}:
        return "val"
    if n in {"test", "testing"}:
        return "test"
    return None


def _discover_coco_files(source: Path) -> tuple[Path, list[tuple[str, Path]]]:
    if source.is_file():
        root = source.parent
        split_guess = _infer_split(source.parent.name)
        if split_guess and source.parent.parent.exists():
            root = source.parent.parent
        files = [p for p in root.rglob("*.json") if "coco" in p.name.lower()]
        if not files and source.suffix.lower() == ".json":
            files = [source]
    else:
        root = source
        files = [p for p in root.rglob("*.json") if "coco" in p.name.lower()]

    split_files: list[tuple[str, Path]] = []
    for path in files:
        split = _infer_split(path.parent.name)
        if split is None:
            split = _infer_split(path.stem) or _infer_split(path.name)
        if split is None:
            continue
        split_files.append((split, path))

    if not split_files:
        raise ValueError("No COCO annotation json files found (expected per-split COCO exports).")

    split_files.sort(key=lambda x: (x[0], str(x[1])))
    return root.resolve(), split_files


def _normalize_segment(coords_abs: list[float], img_w: float, img_h: float) -> list[float]:
    out: list[float] = []
    for i, value in enumerate(coords_abs):
        denom = img_w if i % 2 == 0 else img_h
        if denom <= 0:
            out.append(0.0)
        else:
            out.append(float(value) / float(denom))
    return out


def build_index_from_coco(source: Path, requested_splits: list[str] | None = None, max_images: int = 0) -> BuildIndexResult:
    root, split_files = _discover_coco_files(source)
    include_splits = set(requested_splits or ["train", "val", "test"])

    category_names_by_id: dict[int, str] = {}
    parsed_payloads: list[tuple[str, Path, dict[str, Any]]] = []

    for split, ann_file in split_files:
        if split not in include_splits:
            continue
        payload = json.loads(ann_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            continue
        parsed_payloads.append((split, ann_file, payload))
        for cat in payload.get("categories", []):
            try:
                cid = int(cat.get("id"))
                cname = str(cat.get("name", f"class_{cid}"))
            except Exception:
                continue
            category_names_by_id.setdefault(cid, cname)

    if not parsed_payloads:
        raise ValueError("No COCO split annotation files matched requested splits.")

    sorted_cat_ids = sorted(category_names_by_id)
    class_index_by_cat_id = {cid: idx for idx, cid in enumerate(sorted_cat_ids)}
    class_names = [category_names_by_id[cid] for cid in sorted_cat_ids]

    images_rows: list[dict[str, Any]] = []
    split_stats: dict[str, dict[str, Any]] = {}
    total_seen = 0

    for split, ann_file, payload in sorted(parsed_payloads, key=lambda x: (x[0], str(x[1]))):
        image_items = payload.get("images", [])
        ann_items = payload.get("annotations", [])
        anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for ann in ann_items:
            try:
                image_id = int(ann.get("image_id"))
            except Exception:
                continue
            anns_by_image[image_id].append(ann)

        for image_obj in image_items:
            if max_images > 0 and total_seen >= max_images:
                break
            total_seen += 1

            try:
                image_id = int(image_obj.get("id"))
            except Exception:
                continue

            file_name = str(image_obj.get("file_name", ""))
            img_w = float(image_obj.get("width", 0) or 0)
            img_h = float(image_obj.get("height", 0) or 0)

            image_path = (ann_file.parent / file_name).resolve()
            if not image_path.exists():
                candidate = (ann_file.parent / "images" / file_name).resolve()
                if candidate.exists():
                    image_path = candidate

            try:
                image_rel = image_path.relative_to(root).as_posix()
            except ValueError:
                image_rel = image_path.as_posix()

            if image_path.exists() and image_path.is_file():
                try:
                    stat = image_path.stat()
                    size_bytes = int(stat.st_size)
                    mtime_ns = int(stat.st_mtime_ns)
                    sha256 = _sha256_file(image_path)
                    _, _, image_error = _image_dimensions(image_path)
                except OSError as exc:
                    size_bytes = -1
                    mtime_ns = -1
                    sha256 = ""
                    image_error = str(exc)
            else:
                size_bytes = -1
                mtime_ns = -1
                sha256 = ""
                image_error = "image_not_found"

            label_rows: list[dict[str, Any]] = []
            parse_errors: list[dict[str, Any]] = []

            for ann in anns_by_image.get(image_id, []):
                line = int(ann.get("id", 0) or 0)
                try:
                    category_id = int(ann.get("category_id"))
                except Exception:
                    parse_errors.append({"line": line, "raw": "", "reason": "missing_category_id"})
                    continue
                class_id = class_index_by_cat_id.get(category_id, -1)

                segmentation = ann.get("segmentation")
                used = False
                if isinstance(segmentation, list) and segmentation:
                    seg0 = segmentation[0]
                    if isinstance(seg0, list) and len(seg0) >= 6 and len(seg0) % 2 == 0:
                        try:
                            coords_abs = [float(v) for v in seg0]
                        except Exception:
                            parse_errors.append({"line": line, "raw": "", "reason": "non_numeric_segmentation"})
                            continue
                        coords = _normalize_segment(coords_abs, img_w, img_h)
                        xs = coords[0::2]
                        ys = coords[1::2]
                        x_min, x_max = min(xs), max(xs)
                        y_min, y_max = min(ys), max(ys)
                        label_rows.append(
                            {
                                "line": line,
                                "annotation_type": "segment",
                                "annotation_source": "coco",
                                "class_id": class_id,
                                "x_center": (x_min + x_max) / 2.0,
                                "y_center": (y_min + y_max) / 2.0,
                                "width": x_max - x_min,
                                "height": y_max - y_min,
                                "coords": coords,
                            }
                        )
                        used = True

                if used:
                    continue

                bbox = ann.get("bbox")
                if isinstance(bbox, list) and len(bbox) == 4:
                    try:
                        x, y, w, h = [float(v) for v in bbox]
                    except Exception:
                        parse_errors.append({"line": line, "raw": "", "reason": "non_numeric_bbox"})
                        continue
                    if img_w > 0 and img_h > 0:
                        x_center = (x + (w / 2.0)) / img_w
                        y_center = (y + (h / 2.0)) / img_h
                        width = w / img_w
                        height = h / img_h
                    else:
                        x_center = y_center = width = height = 0.0
                    label_rows.append(
                        {
                            "line": line,
                            "annotation_type": "bbox",
                            "annotation_source": "coco",
                            "class_id": class_id,
                            "x_center": x_center,
                            "y_center": y_center,
                            "width": width,
                            "height": height,
                        }
                    )
                else:
                    parse_errors.append({"line": line, "raw": "", "reason": "missing_bbox_or_segmentation"})

            images_rows.append(
                {
                    "split": split,
                    "image": image_rel,
                    "label": ann_file.relative_to(root).as_posix(),
                    "label_exists": True,
                    "size_bytes": size_bytes,
                    "mtime_ns": mtime_ns,
                    "sha256": sha256,
                    "width": int(img_w) if img_w > 0 else None,
                    "height": int(img_h) if img_h > 0 else None,
                    "image_error": image_error,
                    "annotation_source": "coco",
                    "label_rows": label_rows,
                    "label_parse_errors": parse_errors,
                }
            )

        split_stats.setdefault(
            split,
            {
                "images_dir": ann_file.parent.as_posix(),
                "labels_dir": ann_file.as_posix(),
                "image_count": len(image_items),
                "orphan_labels": [],
            },
        )

    images_rows.sort(key=lambda x: (x["split"], x["image"]))

    cache_basis = []
    for row in images_rows:
        cache_basis.append(f"{row['image']}|{row['size_bytes']}|{row['mtime_ns']}|{row['sha256']}")
    digest = hashlib.sha256("\n".join(cache_basis).encode("utf-8")).hexdigest()

    payload = {
        "schema_version": "1.0.0",
        "dataset_root": root.as_posix(),
        "data_source": source.as_posix(),
        "class_names": class_names,
        "splits": split_stats,
        "images": images_rows,
        "cache_key": f"sha256:{digest}",
    }
    return BuildIndexResult(payload=payload, class_count=len(class_names))
