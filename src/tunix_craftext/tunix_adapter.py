"""Narrow Qwen/Tunix model boundary with explicit local-weight provenance."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import jax

from .llm import LlmRequest, LlmResponse

QWEN_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN_TUNIX_CONFIG_ID = "qwen2.5-0.5b"


def qwen_cache_config(cache_size: int) -> object:
    """Build a Tunix KV cache contract from the pinned Qwen model configuration."""
    if cache_size <= 0:
        raise ValueError("cache_size must be positive")
    from tunix.generate.sampler import CacheConfig
    from tunix.models.automodel import call_model_config

    config = call_model_config(QWEN_TUNIX_CONFIG_ID)
    return CacheConfig(cache_size, config.num_layers, config.num_kv_heads, config.head_dim)


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


def build_qwen_sampler(snapshot: Path, cache_size: int) -> object:
    """Build the public Tunix sampler from explicit local Qwen assets."""
    from tunix.generate.sampler import Sampler

    model, _ = load_qwen_model(snapshot)
    return Sampler(model, load_qwen_tokenizer(snapshot), qwen_cache_config(cache_size))


def build_qwen_single_device_sampler(snapshot: Path, cache_size: int) -> object:
    """Build a working unsharded Tunix sampler for local Qwen smoke inference.

    The named ``fsdp/tp`` path is intentionally not used here: current Tunix
    Qwen generation requires an upstream sharded-gather compatibility fix.
    """
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Qwen snapshot not found: {snapshot}")
    from tunix.generate.sampler import Sampler
    from tunix.models.automodel import call_model_config
    from tunix.models.qwen2.params import create_model_from_safe_tensors

    config = call_model_config(QWEN_TUNIX_CONFIG_ID)
    model = create_model_from_safe_tensors(str(snapshot), config, mesh=None)
    return Sampler(model, load_qwen_tokenizer(snapshot), qwen_cache_config(cache_size))


class QwenTunixBackend:
    """Single-device Tunix Qwen implementation of the typed LLM backend."""

    def __init__(self, snapshot: Path, cache_size: int = 512, seed: int = 0) -> None:
        self._sampler = build_qwen_single_device_sampler(snapshot, cache_size)
        self._cache_size = cache_size
        self._seed = seed

    def complete(self, request: LlmRequest) -> LlmResponse:
        """Generate one raw completion and retain its latency/provenance."""
        started = perf_counter()
        output = self._sampler(
            request.prompt.text,
            max_generation_steps=request.max_new_tokens,
            max_prompt_length=self._cache_size - request.max_new_tokens,
            temperature=request.temperature,
            seed=self._seed,
            return_logprobs=True,
        )
        return LlmResponse(
            raw_text=output.text[0],
            backend="tunix-single-device",
            model=QWEN_MODEL_ID,
            latency_ms=(perf_counter() - started) * 1_000,
        )
