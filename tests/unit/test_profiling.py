"""Tests for lightweight profiling evidence helpers."""

from __future__ import annotations

import json
import time
from pathlib import Path

import jax.numpy as jnp

from tunix_craftext.artifacts.profiling import PhaseProfiler, block_until_ready, save_profile


def test_phase_profiler_records_nested_sections() -> None:
    profiler = PhaseProfiler()

    with profiler.section("prompt"):
        time.sleep(0.001)
        with profiler.section("render"):
            time.sleep(0.001)

    summary = profiler.summary()
    assert set(summary) == {"prompt", "prompt.render"}
    assert summary["prompt"]["calls"] == 1
    assert summary["prompt.render"]["calls"] == 1
    assert summary["prompt"]["total_seconds"] >= summary["prompt.render"]["total_seconds"]


def test_phase_profiler_rejects_ambiguous_section_names() -> None:
    profiler = PhaseProfiler()

    try:
        with profiler.section("llm.generate"):
            pass
    except ValueError as error:
        assert "cannot contain" in str(error)
    else:  # pragma: no cover - defensive clarity for future edits.
        raise AssertionError("expected invalid section name")


def test_block_until_ready_preserves_pytree_values() -> None:
    value = {"x": jnp.asarray([1, 2, 3])}

    synced = block_until_ready(value)

    assert synced["x"].tolist() == [1, 2, 3]


def test_save_profile_writes_stable_json(tmp_path: Path) -> None:
    profiler = PhaseProfiler()
    with profiler.section("env"):
        pass
    path = tmp_path / "profile.json"

    save_profile(path, profiler.events(), metadata={"commit": "test"})

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "tunix-craftext.profile/v1"
    assert payload["metadata"]["commit"] == "test"
    assert payload["events"][0]["name"] == "env"
