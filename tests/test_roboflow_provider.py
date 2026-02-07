from __future__ import annotations

import io
import json
import urllib.error
import zipfile
from pathlib import Path

from dqa.providers.roboflow import resolve_roboflow_data_yaml


class _FakeResponse(io.BytesIO):
    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _zip_with_data_yaml() -> bytes:
    bio = io.BytesIO()
    with zipfile.ZipFile(bio, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "dataset/data.yaml",
            "path: .\ntrain: train/images\nval: valid/images\nnames: [a]\n",
        )
    return bio.getvalue()


def test_roboflow_download_and_extract(monkeypatch, tmp_path: Path) -> None:
    api_payload = {"download": "https://mock.roboflow/download.zip"}
    zip_bytes = _zip_with_data_yaml()
    responses = [
        _FakeResponse(json.dumps(api_payload).encode("utf-8")),
        _FakeResponse(zip_bytes),
    ]

    def _fake_urlopen(_req, timeout=0):  # noqa: ARG001
        return responses.pop(0)

    monkeypatch.setattr("dqa.providers.roboflow.urllib.request.urlopen", _fake_urlopen)
    data_yaml = resolve_roboflow_data_yaml(
        "https://universe.roboflow.com/workspace/project/1",
        tmp_path,
        format_name="yolov11",
        api_key="dummy-key",
    )

    assert data_yaml.exists()
    assert data_yaml.name == "data.yaml"
    text = data_yaml.read_text(encoding="utf-8")
    assert "train: train/images" in text


def test_roboflow_retry_on_transient_api_failure(monkeypatch, tmp_path: Path) -> None:
    api_payload = {"download": "https://mock.roboflow/download.zip"}
    zip_bytes = _zip_with_data_yaml()
    calls = {"n": 0}

    def _fake_urlopen(_req, timeout=0):  # noqa: ARG001
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.URLError("temporary network issue")
        if calls["n"] == 2:
            return _FakeResponse(json.dumps(api_payload).encode("utf-8"))
        return _FakeResponse(zip_bytes)

    monkeypatch.setattr("dqa.providers.roboflow.urllib.request.urlopen", _fake_urlopen)
    monkeypatch.setattr("dqa.providers.roboflow.time.sleep", lambda _x: None)

    data_yaml = resolve_roboflow_data_yaml(
        "https://universe.roboflow.com/workspace/project/1",
        tmp_path,
        format_name="yolov11",
        api_key="dummy-key",
    )
    assert data_yaml.exists()
    assert calls["n"] >= 3


def test_roboflow_cache_hit_skips_network(monkeypatch, tmp_path: Path) -> None:
    extract_dir = tmp_path / "roboflow_workspace_project_1_yolov11" / "extracted" / "dataset"
    extract_dir.mkdir(parents=True, exist_ok=True)
    cached_yaml = extract_dir / "data.yaml"
    cached_yaml.write_text("path: .\ntrain: train/images\nnames: [a]\n", encoding="utf-8")

    def _boom(*_args, **_kwargs):
        raise AssertionError("network call should not happen on cache hit")

    monkeypatch.setattr("dqa.providers.roboflow.urllib.request.urlopen", _boom)

    resolved = resolve_roboflow_data_yaml(
        "https://universe.roboflow.com/workspace/project/1",
        tmp_path,
        format_name="yolov11",
        api_key="dummy-key",
    )
    assert resolved == cached_yaml

def test_roboflow_no_cache_forces_network(monkeypatch, tmp_path: Path) -> None:
    extract_dir = tmp_path / "roboflow_workspace_project_1_yolov11" / "extracted" / "dataset"
    extract_dir.mkdir(parents=True, exist_ok=True)
    (extract_dir / "data.yaml").write_text("path: .\ntrain: train/images\nnames: [a]\n", encoding="utf-8")

    api_payload = {"download": "https://mock.roboflow/download.zip"}
    zip_bytes = _zip_with_data_yaml()
    responses = [
        _FakeResponse(json.dumps(api_payload).encode("utf-8")),
        _FakeResponse(zip_bytes),
    ]
    calls = {"n": 0}

    def _fake_urlopen(_req, timeout=0):  # noqa: ARG001
        calls["n"] += 1
        return responses.pop(0)

    monkeypatch.setattr("dqa.providers.roboflow.urllib.request.urlopen", _fake_urlopen)

    resolved = resolve_roboflow_data_yaml(
        "https://universe.roboflow.com/workspace/project/1",
        tmp_path,
        format_name="yolov11",
        api_key="dummy-key",
        use_cache=False,
    )
    assert resolved.exists()
    assert calls["n"] >= 2


def test_roboflow_ttl_expired_forces_refresh(monkeypatch, tmp_path: Path) -> None:
    extract_dir = tmp_path / "roboflow_workspace_project_1_yolov11" / "extracted" / "dataset"
    extract_dir.mkdir(parents=True, exist_ok=True)
    cached_yaml = extract_dir / "data.yaml"
    cached_yaml.write_text("path: .\ntrain: train/images\nnames: [a]\n", encoding="utf-8")

    api_payload = {"download": "https://mock.roboflow/download.zip"}
    zip_bytes = _zip_with_data_yaml()
    responses = [
        _FakeResponse(json.dumps(api_payload).encode("utf-8")),
        _FakeResponse(zip_bytes),
    ]
    calls = {"n": 0}

    def _fake_urlopen(_req, timeout=0):  # noqa: ARG001
        calls["n"] += 1
        return responses.pop(0)

    monkeypatch.setattr("dqa.providers.roboflow.urllib.request.urlopen", _fake_urlopen)

    resolved = resolve_roboflow_data_yaml(
        "https://universe.roboflow.com/workspace/project/1",
        tmp_path,
        format_name="yolov11",
        api_key="dummy-key",
        cache_ttl_hours=0,
    )
    assert resolved.exists()
    assert calls["n"] >= 2

def test_roboflow_error_includes_payload_keys(monkeypatch, tmp_path: Path) -> None:
    payload = {"foo": "bar", "export": {"status": "not_ready"}}
    def _fake_urlopen(_req, timeout=0):  # noqa: ARG001
        return _FakeResponse(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("dqa.providers.roboflow.urllib.request.urlopen", _fake_urlopen)

    try:
        resolve_roboflow_data_yaml(
            "https://universe.roboflow.com/workspace/project/1",
            tmp_path,
            format_name="coco-segmentation",
            api_key="dummy-key",
            use_cache=False,
        )
    except Exception as exc:
        msg = str(exc)
        assert "Tried formats" in msg
        assert "Observed payload keys" in msg
        assert "foo" in msg
    else:
        raise AssertionError("Expected resolver to fail when no download URL is present")

