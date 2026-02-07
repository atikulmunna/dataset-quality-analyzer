from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from .providers.roboflow import RoboflowProviderError, resolve_roboflow_data_yaml


class RemoteDataError(ValueError):
    """Raised when remote/local data source resolution fails."""


def resolve_data_yaml_source(
    data: Path | None,
    data_url: str | None,
    out_dir: Path,
    data_url_format: str = "yolov11",
    roboflow_api_key: str | None = None,
    use_remote_cache: bool = True,
    remote_cache_ttl_hours: float | None = None,
) -> Path:
    if data and data_url:
        raise RemoteDataError("Use either --data or --data-url, not both.")

    if data:
        if not data.exists():
            raise OSError(f"Data file not found: {data}")
        return data

    if not data_url:
        raise RemoteDataError("One of --data or --data-url is required.")

    parsed = urlparse(data_url)
    if parsed.scheme not in {"http", "https"}:
        raise RemoteDataError("--data-url must be an http(s) URL.")

    host = (parsed.netloc or "").lower()
    if "roboflow.com" in host:
        try:
            return resolve_roboflow_data_yaml(
                data_url,
                out_dir / "_remote",
                format_name=data_url_format,
                api_key=roboflow_api_key,
                use_cache=use_remote_cache,
                cache_ttl_hours=remote_cache_ttl_hours,
            )
        except RoboflowProviderError as exc:
            raise RemoteDataError(str(exc)) from exc

    raise RemoteDataError(
        "Unsupported --data-url provider. Currently supported: Roboflow URLs."
    )

