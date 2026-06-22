"""Narrow Qwen/Tunix model boundary with explicit local-weight provenance."""

from __future__ import annotations

from pathlib import Path

import jax

QWEN_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN_TUNIX_CONFIG_ID = "qwen2.5-0.5b"


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


def load_qwen_model(snapshot: Path) -> tuple[object, Path]:
    """Load Qwen through Tunix on its declared mesh without implicit downloads.

    :param snapshot: Existing local Qwen safetensors snapshot.
    :returns: Tunix NNX model and its resolved snapshot directory.
    :raises FileNotFoundError: If weights have not been explicitly downloaded.
    """
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Qwen snapshot not found: {snapshot}")
    from tunix.models.automodel import AutoModel, ModelSource

    model, resolved = AutoModel.from_pretrained(
        QWEN_MODEL_ID,
        qwen_mesh(),
        model_source=ModelSource.HUGGINGFACE,
        model_download_path=str(snapshot),
    )
    return model, Path(resolved) if resolved is not None else snapshot
