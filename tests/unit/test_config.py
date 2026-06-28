from pathlib import Path

import pytest

from tunix_craftext.env.config import ConfigError, load_mvp_config

ROOT = Path(__file__).resolve().parents[2]


def test_canonical_mvp_config_is_valid_and_reproducible() -> None:
    config = load_mvp_config(ROOT / "configs" / "mvp" / "tiny_craftext.yaml")

    assert config.schema_version == 1
    assert config.environment.batch_size == 2
    assert config.environment.horizon == 8
    assert config.policy.invalid_action == "error"


def test_config_rejects_unknown_schema_fields(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("schema_version: 1\nextra: true\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="keys"):
        load_mvp_config(path)
