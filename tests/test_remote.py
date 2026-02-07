from pathlib import Path

import pytest

from dqa.remote import RemoteDataError, resolve_data_yaml_source


def test_remote_requires_one_source(tmp_path: Path) -> None:
    with pytest.raises(RemoteDataError):
        resolve_data_yaml_source(None, None, tmp_path)


def test_remote_rejects_both_sources(tmp_path: Path) -> None:
    data = tmp_path / "data.yaml"
    data.write_text("path: .\ntrain: train/images\nnames: [a]\n", encoding="utf-8")
    with pytest.raises(RemoteDataError):
        resolve_data_yaml_source(data, "https://example.com", tmp_path)


def test_remote_unsupported_provider(tmp_path: Path) -> None:
    with pytest.raises(RemoteDataError):
        resolve_data_yaml_source(None, "https://example.com/dataset.zip", tmp_path)


def test_remote_roboflow_requires_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(RemoteDataError):
        resolve_data_yaml_source(None, "https://universe.roboflow.com/ws/proj/1", tmp_path)

def test_remote_passes_cache_controls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    expected = tmp_path / "mock" / "data.yaml"
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.write_text("path: .\ntrain: train/images\nnames: [a]\n", encoding="utf-8")

    captured: dict[str, object] = {}

    def _fake_resolver(url, work_dir, format_name="yolov11", api_key=None, use_cache=True, cache_ttl_hours=None):
        captured["url"] = url
        captured["work_dir"] = work_dir
        captured["format_name"] = format_name
        captured["api_key"] = api_key
        captured["use_cache"] = use_cache
        captured["cache_ttl_hours"] = cache_ttl_hours
        return expected

    monkeypatch.setattr("dqa.remote.resolve_roboflow_data_yaml", _fake_resolver)

    resolved = resolve_data_yaml_source(
        None,
        "https://universe.roboflow.com/ws/proj/1",
        tmp_path,
        data_url_format="yolov11",
        roboflow_api_key="abc",
        use_remote_cache=False,
        remote_cache_ttl_hours=12,
    )

    assert resolved == expected
    assert captured["use_cache"] is False
    assert captured["cache_ttl_hours"] == 12
