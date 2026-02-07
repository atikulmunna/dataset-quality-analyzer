from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover
    yaml = None

Severity = Literal["critical", "high", "medium", "low"]


class ConfigError(ValueError):
    """Raised when configuration is invalid."""


@dataclass(frozen=True)
class IntegrityConfig:
    enabled: bool = True


@dataclass(frozen=True)
class ClassDistributionConfig:
    enabled: bool = True
    min_instances_per_class_warn: int = 50
    max_class_share_warn: float = 0.80
    split_drift_jsd_warn: float = 0.10
    split_drift_jsd_high: float = 0.20


@dataclass(frozen=True)
class BBoxSanityConfig:
    enabled: bool = True
    min_box_area_ratio_warn: float = 0.0001
    max_box_area_ratio_warn: float = 0.90
    max_boxes_per_image_warn: int = 300
    aspect_ratio_warn: float = 20.0


@dataclass(frozen=True)
class DuplicatesConfig:
    enabled: bool = True


@dataclass(frozen=True)
class NearDuplicatesConfig:
    enabled: bool = False
    phash_hamming_threshold: int = 8


@dataclass(frozen=True)
class LeakageConfig:
    enabled: bool = True
    include_near_dup: bool = False


@dataclass(frozen=True)
class ChecksConfig:
    integrity: IntegrityConfig
    class_distribution: ClassDistributionConfig
    bbox_sanity: BBoxSanityConfig
    duplicates: DuplicatesConfig
    near_duplicates: NearDuplicatesConfig
    leakage: LeakageConfig


@dataclass(frozen=True)
class DQAConfig:
    version: int
    fail_on: Severity
    checks: ChecksConfig


_EXPECTED_TOP_KEYS = {"version", "fail_on", "checks"}
_EXPECTED_CHECK_KEYS = {
    "integrity",
    "class_distribution",
    "bbox_sanity",
    "duplicates",
    "near_duplicates",
    "leakage",
}
_EXPECTED_SEVERITIES = {"critical", "high", "medium", "low"}


def _expect_dict(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"{name} must be a mapping.")
    return value


def _reject_unknown_keys(name: str, payload: dict[str, Any], expected: set[str]) -> None:
    unknown = sorted(set(payload.keys()) - expected)
    if unknown:
        raise ConfigError(f"{name} has unknown keys: {', '.join(unknown)}")


def _as_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a boolean.")
    return value


def _as_int(name: str, value: Any) -> int:
    if not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer.")
    return value


def _as_float(name: str, value: Any) -> float:
    if not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be numeric.")
    return float(value)


def _parse_integrity(payload: dict[str, Any]) -> IntegrityConfig:
    _reject_unknown_keys("checks.integrity", payload, {"enabled"})
    return IntegrityConfig(enabled=_as_bool("checks.integrity.enabled", payload.get("enabled", True)))


def _parse_class_distribution(payload: dict[str, Any]) -> ClassDistributionConfig:
    _reject_unknown_keys(
        "checks.class_distribution",
        payload,
        {
            "enabled",
            "min_instances_per_class_warn",
            "max_class_share_warn",
            "split_drift_jsd_warn",
            "split_drift_jsd_high",
        },
    )
    return ClassDistributionConfig(
        enabled=_as_bool("checks.class_distribution.enabled", payload.get("enabled", True)),
        min_instances_per_class_warn=_as_int(
            "checks.class_distribution.min_instances_per_class_warn",
            payload.get("min_instances_per_class_warn", 50),
        ),
        max_class_share_warn=_as_float(
            "checks.class_distribution.max_class_share_warn", payload.get("max_class_share_warn", 0.80)
        ),
        split_drift_jsd_warn=_as_float(
            "checks.class_distribution.split_drift_jsd_warn", payload.get("split_drift_jsd_warn", 0.10)
        ),
        split_drift_jsd_high=_as_float(
            "checks.class_distribution.split_drift_jsd_high", payload.get("split_drift_jsd_high", 0.20)
        ),
    )


def _parse_bbox_sanity(payload: dict[str, Any]) -> BBoxSanityConfig:
    _reject_unknown_keys(
        "checks.bbox_sanity",
        payload,
        {
            "enabled",
            "min_box_area_ratio_warn",
            "max_box_area_ratio_warn",
            "max_boxes_per_image_warn",
            "aspect_ratio_warn",
        },
    )
    return BBoxSanityConfig(
        enabled=_as_bool("checks.bbox_sanity.enabled", payload.get("enabled", True)),
        min_box_area_ratio_warn=_as_float(
            "checks.bbox_sanity.min_box_area_ratio_warn", payload.get("min_box_area_ratio_warn", 0.0001)
        ),
        max_box_area_ratio_warn=_as_float(
            "checks.bbox_sanity.max_box_area_ratio_warn", payload.get("max_box_area_ratio_warn", 0.90)
        ),
        max_boxes_per_image_warn=_as_int(
            "checks.bbox_sanity.max_boxes_per_image_warn", payload.get("max_boxes_per_image_warn", 300)
        ),
        aspect_ratio_warn=_as_float("checks.bbox_sanity.aspect_ratio_warn", payload.get("aspect_ratio_warn", 20.0)),
    )


def _parse_duplicates(payload: dict[str, Any]) -> DuplicatesConfig:
    _reject_unknown_keys("checks.duplicates", payload, {"enabled"})
    return DuplicatesConfig(enabled=_as_bool("checks.duplicates.enabled", payload.get("enabled", True)))


def _parse_near_duplicates(payload: dict[str, Any]) -> NearDuplicatesConfig:
    _reject_unknown_keys("checks.near_duplicates", payload, {"enabled", "phash_hamming_threshold"})
    return NearDuplicatesConfig(
        enabled=_as_bool("checks.near_duplicates.enabled", payload.get("enabled", False)),
        phash_hamming_threshold=_as_int(
            "checks.near_duplicates.phash_hamming_threshold", payload.get("phash_hamming_threshold", 8)
        ),
    )


def _parse_leakage(payload: dict[str, Any]) -> LeakageConfig:
    _reject_unknown_keys("checks.leakage", payload, {"enabled", "include_near_dup"})
    return LeakageConfig(
        enabled=_as_bool("checks.leakage.enabled", payload.get("enabled", True)),
        include_near_dup=_as_bool("checks.leakage.include_near_dup", payload.get("include_near_dup", False)),
    )


def load_config(path: Path) -> DQAConfig:
    if yaml is None:
        raise ConfigError("PyYAML is required to load dqa.yaml. Install with: pip install pyyaml")
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _expect_dict("config", raw or {})

    _reject_unknown_keys("config", root, _EXPECTED_TOP_KEYS)

    version = _as_int("version", root.get("version", 1))
    fail_on = root.get("fail_on", "high")
    if fail_on not in _EXPECTED_SEVERITIES:
        raise ConfigError("fail_on must be one of: critical, high, medium, low")

    checks_raw = _expect_dict("checks", root.get("checks", {}))
    _reject_unknown_keys("checks", checks_raw, _EXPECTED_CHECK_KEYS)

    checks = ChecksConfig(
        integrity=_parse_integrity(_expect_dict("checks.integrity", checks_raw.get("integrity", {}))),
        class_distribution=_parse_class_distribution(
            _expect_dict("checks.class_distribution", checks_raw.get("class_distribution", {}))
        ),
        bbox_sanity=_parse_bbox_sanity(_expect_dict("checks.bbox_sanity", checks_raw.get("bbox_sanity", {}))),
        duplicates=_parse_duplicates(_expect_dict("checks.duplicates", checks_raw.get("duplicates", {}))),
        near_duplicates=_parse_near_duplicates(
            _expect_dict("checks.near_duplicates", checks_raw.get("near_duplicates", {}))
        ),
        leakage=_parse_leakage(_expect_dict("checks.leakage", checks_raw.get("leakage", {}))),
    )

    return DQAConfig(version=version, fail_on=fail_on, checks=checks)
