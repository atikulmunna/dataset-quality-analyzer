from __future__ import annotations

import signal

import pytest

from dqa.worker_entry import _terminate


def test_worker_sigterm_uses_conventional_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _terminate(signal.SIGTERM, None)

    assert exc_info.value.code == 128 + signal.SIGTERM
    assert "SIGTERM" in capsys.readouterr().err

