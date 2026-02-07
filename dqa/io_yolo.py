from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None


class DatasetSpecError(ValueError):
    """Raised when dataset layout/config cannot be resolved."""


@dataclass(frozen=True)
class SplitPaths:
    name: str
    images_dir: Path
    labels_dir: Path


@dataclass(frozen=True)
class DatasetSpec:
    data_yaml: Path
    root: Path
    names: list[str]
    splits: dict[str, SplitPaths]


def _ensure_mapping(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DatasetSpecError(f"{name} must be a mapping")
    return value


def _resolve_names(raw_names: Any) -> list[str]:
    if isinstance(raw_names, list) and all(isinstance(v, str) for v in raw_names):
        return list(raw_names)
    if isinstance(raw_names, dict):
        parsed: list[tuple[int, str]] = []
        for key, value in raw_names.items():
            try:
                idx = int(key)
            except (TypeError, ValueError) as exc:
                raise DatasetSpecError("data.yaml names keys must be numeric") from exc
            if not isinstance(value, str):
                raise DatasetSpecError("data.yaml names values must be strings")
            parsed.append((idx, value))
        parsed.sort(key=lambda x: x[0])
        return [name for _, name in parsed]
    raise DatasetSpecError("data.yaml names must be a list or mapping")


def _resolve_path(base: Path, maybe_relative: str) -> Path:
    path = Path(maybe_relative)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def _labels_dir_for_images(images_dir: Path) -> Path:
    if images_dir.name.lower() == "images":
        return images_dir.parent / "labels"
    return images_dir.parent / "labels"


def load_dataset_spec(data_yaml_path: Path, requested_splits: list[str] | None = None) -> DatasetSpec:
    if yaml is None:
        raise DatasetSpecError("PyYAML is required to parse data.yaml. Install with: pip install pyyaml")
    if not data_yaml_path.exists():
        raise DatasetSpecError(f"data.yaml not found: {data_yaml_path}")

    raw = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    data = _ensure_mapping("data.yaml", raw or {})

    root_hint = data.get("path", ".")
    if not isinstance(root_hint, str):
        raise DatasetSpecError("data.yaml path must be a string")
    root = _resolve_path(data_yaml_path.parent, root_hint)

    names = _resolve_names(data.get("names"))

    allowed = ["train", "val", "test"]
    splits_to_use = requested_splits or allowed

    splits: dict[str, SplitPaths] = {}
    for split_name in splits_to_use:
        if split_name not in allowed:
            raise DatasetSpecError(f"Unsupported split requested: {split_name}")
        split_value = data.get(split_name)
        if split_value is None:
            continue
        if not isinstance(split_value, str):
            raise DatasetSpecError(f"data.yaml {split_name} must be a directory path string")
        images_dir = _resolve_path(root, split_value)
        labels_dir = _labels_dir_for_images(images_dir)
        splits[split_name] = SplitPaths(name=split_name, images_dir=images_dir, labels_dir=labels_dir)

    if not splits:
        raise DatasetSpecError("No valid splits were resolved from data.yaml")

    return DatasetSpec(data_yaml=data_yaml_path.resolve(), root=root, names=names, splits=splits)
