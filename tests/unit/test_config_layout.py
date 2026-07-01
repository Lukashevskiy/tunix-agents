"""Repository-level contracts for the canonical config tree."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_INDEX = ROOT / "configs/index.yaml"


def _leaf_paths(value: object) -> list[str]:
    """Return every YAML scalar path value from nested config index mappings."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        paths: list[str] = []
        for child in value.values():
            paths.extend(_leaf_paths(child))
        return paths
    return []


def test_config_index_points_to_existing_canonical_files() -> None:
    """Every canonical config path in the index must resolve to a file."""
    index = cast(dict[str, Any], yaml.safe_load(CONFIG_INDEX.read_text(encoding="utf-8")))
    canonical = cast(dict[str, Any], index["canonical_layout"])

    paths = _leaf_paths(canonical)

    assert paths
    assert all(path.startswith("configs/") for path in paths)
    assert all((ROOT / path).is_file() for path in paths)


def test_legacy_config_paths_are_symlink_aliases_to_canonical_layout() -> None:
    """Legacy paths stay usable, but canonical paths remain the source of truth."""
    index = cast(dict[str, Any], yaml.safe_load(CONFIG_INDEX.read_text(encoding="utf-8")))
    aliases = cast(dict[str, str], index["compatibility_aliases"])

    for legacy, canonical in aliases.items():
        legacy_path = ROOT / legacy
        canonical_path = ROOT / canonical

        assert canonical_path.is_file(), canonical
        assert legacy_path.is_symlink(), legacy
        assert legacy_path.resolve() == canonical_path.resolve()
