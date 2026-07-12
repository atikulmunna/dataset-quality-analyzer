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


@pytest.mark.parametrize("name", ["dqa.yaml", "dqa_seg.yaml", "dqa_seg_low_noise.yaml"])
def test_repository_presets_load(name: str) -> None:
    config = load_config(Path(name))

    assert config.version == 1
    assert config.fail_on == "high"


def test_builtin_defaults_match_detection_preset() -> None:
    assert load_config() == load_config(Path("dqa.yaml"))


def test_removed_near_leakage_option_is_rejected(tmp_path: Path) -> None:
    cfg = tmp_path / "dqa.yaml"
    cfg.write_text(
        """
version: 1
fail_on: high
checks:
  leakage:
    enabled: true
    include_near_dup: true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="include_near_dup"):
        load_config(cfg)
