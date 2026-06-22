"""Stable Qwen/Tunix model adapter contracts independent of loaded weights."""

from __future__ import annotations

import pytest

from tunix_craftext.tunix_adapter import (
    QWEN_MODEL_ID,
    QWEN_TUNIX_CONFIG_ID,
    load_qwen_model,
    load_qwen_tokenizer,
    qwen_cache_config,
    qwen_mesh,
)


def test_qwen_repo_and_tunix_config_ids_are_distinct_and_explicit() -> None:
    """Tunix naming must never infer a config id from a Hugging Face repo id."""
    assert QWEN_MODEL_ID == "Qwen/Qwen2.5-0.5B-Instruct"
    assert QWEN_TUNIX_CONFIG_ID == "qwen2.5-0.5b"


def test_qwen_mesh_exposes_required_tunix_axes() -> None:
    """Qwen's Tunix sharding template requires fsdp and tp, even on one device."""
    assert qwen_mesh().axis_names == ("fsdp", "tp")


def test_qwen_cache_config_uses_pinned_model_dimensions() -> None:
    """Sampler cache tracks the exact Qwen KV layout, not guessed constants."""
    cache = qwen_cache_config(128)
    assert (cache.cache_size, cache.num_layers, cache.num_kv_heads, cache.head_dim) == (
        128,
        24,
        2,
        64,
    )

def test_qwen_tokenizer_requires_explicit_local_snapshot(tmp_path) -> None:
    """No test or library import may silently download model assets."""
    with pytest.raises(FileNotFoundError):
        load_qwen_tokenizer(tmp_path / QWEN_MODEL_ID)


def test_qwen_model_requires_explicit_local_snapshot(tmp_path) -> None:
    """Model loading never falls through to an implicit network download."""
    with pytest.raises(FileNotFoundError):
        load_qwen_model(tmp_path / QWEN_MODEL_ID)
