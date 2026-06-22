"""Narrow Qwen/Tunix model boundary with explicit local-weight provenance."""

from __future__ import annotations

from pathlib import Path

import jax

QWEN_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def qwen_mesh() -> jax.sharding.Mesh:
    """Return the single-device-compatible Tunix Qwen mesh with required axes."""
    return jax.make_mesh((1, 1), ("fsdp", "tp"))


def load_qwen_tokenizer(snapshot: Path) -> object:
    """Load a Tunix HF tokenizer from an already downloaded Qwen snapshot.

    :param snapshot: Local Hugging Face snapshot directory.
    :returns: Tunix tokenizer adapter with ``encode`` and ``decode``.
    :raises FileNotFoundError: If a local snapshot has not been explicitly downloaded.
    """
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Qwen snapshot not found: {snapshot}")
    from tunix.generate.tokenizer_adapter import Tokenizer

    return Tokenizer(
        tokenizer_type="huggingface", tokenizer_path=str(snapshot), add_bos=False, add_eos=False
    )
