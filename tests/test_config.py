from pathlib import Path

import pytest

from dqa.config import ConfigError, load_config


def test_config_unknown_key_raises(tmp_path: Path) -> None:
    cfg = tmp_path / "dqa.yaml"
    cfg.write_text(
        """
version: 1
fail_on: high
unknown_key: true
checks:
  integrity:
    enabled: true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError):
        load_config(cfg)
