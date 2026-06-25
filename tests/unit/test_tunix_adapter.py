"""Stable Qwen/Tunix model adapter contracts independent of loaded weights."""

from __future__ import annotations

import pytest

from tunix_craftext.tunix_adapter import (
    GEMMA_ACTION_SYSTEM_PROMPT,
    GEMMA_MODEL_ID,
    GEMMA_TUNIX_CONFIG_ID,
    QWEN_MODEL_ID,
    QWEN_TUNIX_CONFIG_ID,
    format_gemma_action_prompt,
    gemma_cache_config,
    load_gemma_single_device_model,
    load_gemma_tokenizer,
    load_qwen_model,
    load_qwen_tokenizer,
    qwen_cache_config,
    qwen_mesh,
)


def test_qwen_repo_and_tunix_config_ids_are_distinct_and_explicit() -> None:
    """Tunix naming must never infer a config id from a Hugging Face repo id."""
    assert QWEN_MODEL_ID == "Qwen/Qwen2.5-0.5B-Instruct"
    assert QWEN_TUNIX_CONFIG_ID == "qwen2.5-0.5b"


def test_gemma_repo_and_tunix_config_ids_are_distinct_and_explicit() -> None:
    """Gemma HF id and Tunix config id are pinned separately."""
    assert GEMMA_MODEL_ID == "google/gemma-3-270m-it"
    assert GEMMA_TUNIX_CONFIG_ID == "gemma3_270m_it"


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


def test_gemma_cache_config_uses_pinned_model_dimensions() -> None:
    """Sampler cache tracks Gemma3 KV layout from Tunix, not guessed constants."""
    cache = gemma_cache_config(128)
    assert cache.cache_size == 128
    assert cache.num_layers > 0
    assert cache.num_kv_heads > 0
    assert cache.head_dim > 0


class _ChatTokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        del add_generation_prompt, tokenize
        assert messages == [
            {"role": "user", "content": f"{GEMMA_ACTION_SYSTEM_PROMPT}\n\nchoose"}
        ]
        return "<start_of_turn>user\nchoose<end_of_turn>\n<start_of_turn>model\n"


def test_format_gemma_action_prompt_uses_gemma_user_turn() -> None:
    """Gemma has no separate system role in the Tunix SP template path."""
    rendered = format_gemma_action_prompt(_ChatTokenizer(), "choose")

    assert rendered.startswith("<start_of_turn>user")
    assert rendered.endswith("<start_of_turn>model\n")


def test_qwen_tokenizer_requires_explicit_local_snapshot(tmp_path) -> None:
    """No test or library import may silently download model assets."""
    with pytest.raises(FileNotFoundError):
        load_qwen_tokenizer(tmp_path / QWEN_MODEL_ID)


def test_gemma_tokenizer_requires_explicit_local_snapshot(tmp_path) -> None:
    """Gemma tokenizer loading never falls through to a remote/default path."""
    with pytest.raises(FileNotFoundError):
        load_gemma_tokenizer(tmp_path / GEMMA_MODEL_ID)


def test_qwen_model_requires_explicit_local_snapshot(tmp_path) -> None:
    """Model loading never falls through to an implicit network download."""
    with pytest.raises(FileNotFoundError):
        load_qwen_model(tmp_path / QWEN_MODEL_ID)


def test_gemma_model_requires_explicit_local_snapshot(tmp_path) -> None:
    """Gemma model loading never falls through to an implicit network download."""
    with pytest.raises(FileNotFoundError):
        load_gemma_single_device_model(tmp_path / GEMMA_MODEL_ID)
