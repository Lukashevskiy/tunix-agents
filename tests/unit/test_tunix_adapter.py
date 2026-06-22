"""Stable Qwen/Tunix model adapter contracts independent of loaded weights."""

from __future__ import annotations

import pytest

from tunix_craftext.tunix_adapter import QWEN_MODEL_ID, load_qwen_tokenizer, qwen_mesh


def test_qwen_mesh_exposes_required_tunix_axes() -> None:
    """Qwen's Tunix sharding template requires fsdp and tp, even on one device."""
    assert qwen_mesh().axis_names == ("fsdp", "tp")


def test_qwen_tokenizer_requires_explicit_local_snapshot(tmp_path) -> None:
    """No test or library import may silently download model assets."""
    with pytest.raises(FileNotFoundError):
        load_qwen_tokenizer(tmp_path / QWEN_MODEL_ID)
