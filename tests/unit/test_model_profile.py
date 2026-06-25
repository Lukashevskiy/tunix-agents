"""Tests for strict LLM backbone model profiles."""

from __future__ import annotations

from pathlib import Path

import pytest

from tunix_craftext.model_profile import ModelProfileError, load_model_profile

ROOT = Path(__file__).resolve().parents[2]


def test_gemma_270m_profile_is_llm_actor_candidate() -> None:
    profile = load_model_profile(ROOT / "configs/models/gemma3_270m_instruction.yaml")

    assert profile.architecture == "gemma3"
    assert profile.model_id == "google/gemma-3-270m-it"
    assert profile.source == "gcs"
    assert profile.purpose == "tunix-llm-actor-backbone"
    assert profile.is_llm_actor_candidate


def test_qwen_profile_is_llm_actor_candidate() -> None:
    profile = load_model_profile(ROOT / "configs/models/qwen25_05b_instruction.yaml")

    assert profile.architecture == "qwen2"
    assert profile.is_llm_actor_candidate


def test_model_profile_rejects_schema_drift(tmp_path: Path) -> None:
    invalid = tmp_path / "bad.yaml"
    invalid.write_text(
        (ROOT / "configs/models/gemma3_270m_instruction.yaml").read_text(encoding="utf-8")
        + "\nextra: nope\n",
        encoding="utf-8",
    )

    with pytest.raises(ModelProfileError, match="root keys"):
        load_model_profile(invalid)
