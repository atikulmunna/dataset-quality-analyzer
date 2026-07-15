from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile

from scripts.build_lambda import build


def test_lambda_package_is_deterministic_and_runtime_scoped(tmp_path: Path) -> None:
    first = tmp_path / "first.zip"
    second = tmp_path / "second.zip"
    build(first)
    build(second)

    assert hashlib.sha256(first.read_bytes()).digest() == hashlib.sha256(second.read_bytes()).digest()
    with zipfile.ZipFile(first) as archive:
        names = set(archive.namelist())
    assert "dqa/aws/api_handler.py" in names
    assert "dqa/aws/cost_guard.py" in names
    assert "dqa/web/api.py" in names
    assert "dqa/audit.py" not in names
    assert "dqa/aws/worker.py" not in names

