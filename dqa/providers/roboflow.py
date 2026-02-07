from __future__ import annotations

import json
import os
import re
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path


class RoboflowProviderError(ValueError):
    """Raised for Roboflow provider issues."""


_MAX_RETRIES = 3
_BACKOFF_SECONDS = 0.8


def _parse_reference(url: str) -> tuple[str, str, str]:
    parsed = urllib.parse.urlparse(url)
    parts = [p for p in parsed.path.split("/") if p]
    if len(parts) < 3:
        raise RoboflowProviderError(
            "Roboflow URL must include workspace/project/version, "
            "for example: https://universe.roboflow.com/ws/proj/1"
        )

    workspace = parts[0]
    project = parts[1]
    if len(parts) >= 4 and parts[2].lower() == "dataset":
        version = parts[3]
    else:
        version = parts[2]

    if not re.match(r"^[A-Za-z0-9._-]+$", workspace):
        raise RoboflowProviderError("Could not parse workspace from Roboflow URL.")
    if not re.match(r"^[A-Za-z0-9._-]+$", project):
        raise RoboflowProviderError("Could not parse project from Roboflow URL.")
    if not re.match(r"^[A-Za-z0-9._-]+$", version):
        raise RoboflowProviderError("Could not parse version from Roboflow URL.")
    return workspace, project, version


def _http_json(url: str, timeout_sec: float = 30.0) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "DQA/0.1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            data = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise RoboflowProviderError(f"Roboflow API HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RoboflowProviderError(f"Roboflow API connection failed: {exc.reason}") from exc
    try:
        payload = json.loads(data)
    except json.JSONDecodeError as exc:
        raise RoboflowProviderError("Roboflow API returned non-JSON response.") from exc
    if not isinstance(payload, dict):
        raise RoboflowProviderError("Roboflow API response was not a JSON object.")
    return payload


def _http_json_with_retry(url: str, timeout_sec: float = 30.0, retries: int = _MAX_RETRIES) -> dict:
    last_error: RoboflowProviderError | None = None
    for attempt in range(1, retries + 1):
        try:
            return _http_json(url, timeout_sec=timeout_sec)
        except RoboflowProviderError as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(_BACKOFF_SECONDS * attempt)
    if last_error:
        raise last_error
    raise RoboflowProviderError("Unknown error while fetching Roboflow API JSON.")


def _find_first_url(value: object) -> str | None:
    if isinstance(value, str):
        low = value.lower()
        if low.startswith("http://") or low.startswith("https://"):
            return value
        return None
    if isinstance(value, dict):
        for candidate_key in ("download", "url", "export", "export_url", "download_url"):
            if candidate_key in value:
                found = _find_first_url(value[candidate_key])
                if found:
                    return found
        for child in value.values():
            found = _find_first_url(child)
            if found:
                return found
        return None
    if isinstance(value, list):
        for child in value:
            found = _find_first_url(child)
            if found:
                return found
    return None


def _format_aliases(format_name: str) -> list[str]:
    normalized = (format_name or "").strip().lower()
    if not normalized:
        return ["yolov11"]

    aliases: list[str] = [normalized]

    if "coco" in normalized:
        aliases.extend([
            "coco",
            "coco-segmentation",
            "coco-json",
            "coco-seg",
            "coco-instance-segmentation",
            "coco-segmentation-json",
        ])

    if "semantic" in normalized or "mask" in normalized:
        aliases.extend([
            "semantic-segmentation-mask",
            "semantic-segmentation",
        ])

    dedup: list[str] = []
    seen: set[str] = set()
    for item in aliases:
        if item not in seen:
            seen.add(item)
            dedup.append(item)
    return dedup


def _payload_keys(payload: dict[str, object]) -> str:
    keys = sorted(payload.keys())
    return ",".join(keys[:12]) if keys else "<none>"


def _resolve_export_url(workspace: str, project: str, version: str, format_name: str, api_key: str) -> str:
    params = urllib.parse.urlencode({"api_key": api_key})

    tried_formats = _format_aliases(format_name)
    inspected_payload_keys: list[str] = []
    endpoint_failures: list[str] = []

    last_error: RoboflowProviderError | None = None

    for fmt in tried_formats:
        endpoint_candidates = [
            f"https://api.roboflow.com/{workspace}/{project}/{version}/{fmt}?{params}",
            f"https://api.roboflow.com/dataset/{workspace}/{project}/{version}/{fmt}?{params}",
        ]

        for endpoint in endpoint_candidates:
            try:
                payload = _http_json_with_retry(endpoint)
            except RoboflowProviderError as exc:
                last_error = exc
                endpoint_failures.append(f"{fmt}:{endpoint} -> {exc}")
                continue

            inspected_payload_keys.append(f"{fmt}:[{_payload_keys(payload)}]")
            found = _find_first_url(payload)
            if found:
                return found

    if inspected_payload_keys:
        raise RoboflowProviderError(
            "Could not resolve export download URL from Roboflow response. "
            f"Tried formats={tried_formats}. "
            f"Observed payload keys={'; '.join(inspected_payload_keys[:6])}."
        )

    if last_error:
        raise RoboflowProviderError(
            f"Could not resolve export URL after trying formats={tried_formats}. "
            f"Last error: {last_error}"
        ) from last_error

    raise RoboflowProviderError(
        f"Could not resolve export URL. Tried formats={tried_formats}."
    )


def _download_file(url: str, dest: Path, timeout_sec: float = 120.0) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "DQA/0.1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as out:
                shutil.copyfileobj(resp, out)
    except urllib.error.HTTPError as exc:
        raise RoboflowProviderError(f"Dataset download HTTP {exc.code}: {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise RoboflowProviderError(f"Dataset download failed: {exc.reason}") from exc


def _download_file_with_retry(url: str, dest: Path, timeout_sec: float = 120.0, retries: int = _MAX_RETRIES) -> None:
    last_error: RoboflowProviderError | None = None
    for attempt in range(1, retries + 1):
        try:
            _download_file(url, dest, timeout_sec=timeout_sec)
            return
        except RoboflowProviderError as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(_BACKOFF_SECONDS * attempt)
    if last_error:
        raise last_error
    raise RoboflowProviderError("Unknown error while downloading dataset archive.")


def _find_data_yaml(root: Path) -> Path:
    candidates = sorted(p for p in root.rglob("data.yaml") if p.is_file())
    if not candidates:
        raise RoboflowProviderError("Downloaded archive does not contain data.yaml.")
    candidates.sort(key=lambda p: (len(p.parts), str(p)))
    return candidates[0]


def _is_cache_fresh(data_yaml: Path, ttl_hours: float | None) -> bool:
    if ttl_hours is None:
        return True
    if ttl_hours <= 0:
        return False
    age_sec = time.time() - data_yaml.stat().st_mtime
    return age_sec <= ttl_hours * 3600.0


def resolve_roboflow_data_yaml(
    url: str,
    work_dir: Path,
    format_name: str = "yolov11",
    api_key: str | None = None,
    use_cache: bool = True,
    cache_ttl_hours: float | None = None,
) -> Path:
    """Resolve Roboflow URL into a local data.yaml path."""
    api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RoboflowProviderError(
            "ROBOFLOW_API_KEY is required for Roboflow URL ingestion. "
            "Set the env var or provide --data with a local data.yaml path."
        )

    workspace, project, version = _parse_reference(url)

    run_dir = work_dir / f"roboflow_{workspace}_{project}_{version}_{format_name}"
    zip_path = run_dir / "dataset.zip"
    extract_dir = run_dir / "extracted"

    if use_cache and extract_dir.exists():
        try:
            cached_yaml = _find_data_yaml(extract_dir)
            if _is_cache_fresh(cached_yaml, cache_ttl_hours):
                return cached_yaml
        except RoboflowProviderError:
            pass

    download_url = _resolve_export_url(workspace, project, version, format_name, api_key)
    _download_file_with_retry(download_url, zip_path)
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
    except zipfile.BadZipFile as exc:
        raise RoboflowProviderError("Downloaded artifact is not a valid ZIP archive.") from exc

    return _find_data_yaml(extract_dir)
