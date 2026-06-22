"""Узкая граница модели Qwen/Tunix с явным происхождением локальных весов."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Protocol, cast

import jax

from .llm import LlmRequest, LlmResponse

QWEN_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
QWEN_TUNIX_CONFIG_ID = "qwen2.5-0.5b"


class ChatTemplatingTokenizer(Protocol):
    """Public tokenizer capability required to prepare a Qwen assistant turn."""

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str | list[int]: ...


class SamplingTokenizer(ChatTemplatingTokenizer, Protocol):
    """Tokenizer operations needed to size a static Tunix sampler cache."""

    def encode(self, text: str) -> list[int]: ...

    def bos_id(self) -> int: ...

    def pad_id(self) -> int: ...


class CacheConfigLike(Protocol):
    """Stable cache dimensions consumed by the Qwen sampler boundary."""

    cache_size: int
    num_layers: int
    num_kv_heads: int
    head_dim: int


class SamplerOutputLike(Protocol):
    """Public sampler result fields retained by the LLM boundary."""

    text: list[str]
    logprobs: list[list[float]] | None
    tokens: list[Sequence[int]]
    padded_prompt_tokens: Sequence[Sequence[int]]


class TextSampler(Protocol):
    """Subset of Tunix sampler API required by one text completion."""

    def __call__(
        self,
        input_strings: str,
        *,
        max_generation_steps: int,
        max_prompt_length: int,
        temperature: float,
        seed: int,
        return_logprobs: bool,
    ) -> SamplerOutputLike: ...


class HiddenStateModel(Protocol):
    """Qwen model subset used to extract token features without the language head."""

    def __call__(
        self,
        input_tokens: jax.Array,
        positions: jax.Array,
        cache: object | None,
        attention_mask: jax.Array | None,
        *,
        skip_lm_head: bool,
    ) -> tuple[jax.Array, object | None]: ...


QWEN_ACTION_SYSTEM_PROMPT = (
    "You are a CrafText agent. Choose exactly one action from the action catalogue in the "
    "user message. Your complete response is one XML action element containing that exact action "
    "name. Never output the placeholder word LABEL and do not add an explanation."
)


def format_qwen_action_prompt(
    tokenizer: ChatTemplatingTokenizer, prompt_text: str
) -> str:
    """Apply Qwen's declared chat template to one already-rendered action prompt.

    :raises ValueError: If the tokenizer does not produce textual chat input.
    """
    if not prompt_text.strip():
        raise ValueError("prompt_text must be non-empty")
    rendered = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": QWEN_ACTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    if not isinstance(rendered, str) or not rendered.strip():
        raise ValueError("Qwen chat template must return non-empty text")
    return rendered


def qwen_required_cache_size(
    tokenizer: SamplingTokenizer, chat_prompt: str, max_new_tokens: int
) -> int:
    """Return Tunix's padded prompt requirement plus requested completion tokens."""
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    input_tokens = len(tokenizer.encode(chat_prompt)) + int(bool(tokenizer.bos_id()))
    padded_prompt_tokens = 1 << max(input_tokens - 1, 0).bit_length()
    return padded_prompt_tokens + max_new_tokens


def qwen_cache_config(cache_size: int) -> CacheConfigLike:
    """Build a Tunix KV cache contract from the pinned Qwen model configuration."""
    if cache_size <= 0:
        raise ValueError("cache_size must be positive")
    from tunix.generate.sampler import CacheConfig  # type: ignore[import-untyped]
    from tunix.models.automodel import call_model_config  # type: ignore[import-untyped]

    config = call_model_config(QWEN_TUNIX_CONFIG_ID)
    return cast(
        CacheConfigLike,
        CacheConfig(cache_size, config.num_layers, config.num_kv_heads, config.head_dim),
    )


def qwen_mesh() -> jax.sharding.Mesh:
    """Return the single-device-compatible Tunix Qwen mesh with required axes."""
    return jax.make_mesh((1, 1), ("fsdp", "tp"))


def load_qwen_tokenizer(snapshot: Path) -> SamplingTokenizer:
    """Load a Tunix HF tokenizer from an already downloaded Qwen snapshot.

    :param snapshot: Local Hugging Face snapshot directory.
    :returns: Tunix tokenizer adapter with ``encode`` and ``decode``.
    :raises FileNotFoundError: If a local snapshot has not been explicitly downloaded.
    """
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Qwen snapshot not found: {snapshot}")
    from tunix.generate.tokenizer_adapter import Tokenizer  # type: ignore[import-untyped]

    return cast(
        SamplingTokenizer,
        Tokenizer(
            tokenizer_type="huggingface",
            tokenizer_path=str(snapshot),
            add_bos=False,
            add_eos=False,
        ),
    )


def load_qwen_model(snapshot: Path) -> tuple[object, Path]:
    """Load Qwen through Tunix on its declared mesh without implicit downloads.

    :param snapshot: Existing local Qwen safetensors snapshot.
    :returns: Tunix NNX model and its resolved snapshot directory.
    :raises FileNotFoundError: If weights have not been explicitly downloaded.
    """
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Qwen snapshot not found: {snapshot}")
    from tunix.models.automodel import AutoModel, ModelSource  # type: ignore[import-untyped]

    model, resolved = AutoModel.from_pretrained(
        QWEN_MODEL_ID,
        qwen_mesh(),
        model_source=ModelSource.HUGGINGFACE,
        model_download_path=str(snapshot),
    )
    return model, Path(resolved) if resolved is not None else snapshot


def load_qwen_single_device_model(snapshot: Path) -> HiddenStateModel:
    """Load local Qwen weights once for single-device sampling and feature extraction."""
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Qwen snapshot not found: {snapshot}")
    from tunix.models.automodel import call_model_config  # type: ignore[import-untyped]
    from tunix.models.qwen2.params import (  # type: ignore[import-untyped]
        create_model_from_safe_tensors,
    )

    config = call_model_config(QWEN_TUNIX_CONFIG_ID)
    return cast(HiddenStateModel, create_model_from_safe_tensors(str(snapshot), config, mesh=None))


def qwen_chat_token_ids(tokenizer: SamplingTokenizer, chat_prompt: str) -> tuple[int, ...]:
    """Encode one chat prompt exactly as the Tunix sampler does before padding."""
    bos = (tokenizer.bos_id(),) if tokenizer.bos_id() else ()
    return (*bos, *(int(token) for token in tokenizer.encode(chat_prompt)))


def build_qwen_sampler(snapshot: Path, cache_size: int) -> TextSampler:
    """Build the public Tunix sampler from explicit local Qwen assets."""
    from tunix.generate.sampler import Sampler  # type: ignore[import-untyped]

    model, _ = load_qwen_model(snapshot)
    sampler = Sampler(model, load_qwen_tokenizer(snapshot), qwen_cache_config(cache_size))
    return cast(TextSampler, sampler)


def build_qwen_single_device_sampler(
    snapshot: Path,
    cache_size: int,
    tokenizer: SamplingTokenizer | None = None,
    model: HiddenStateModel | None = None,
) -> TextSampler:
    """Build a working unsharded Tunix sampler for local Qwen smoke inference.

    The named ``fsdp/tp`` path is intentionally not used here: current Tunix
    Qwen generation requires an upstream sharded-gather compatibility fix.
    """
    if not snapshot.is_dir():
        raise FileNotFoundError(f"Qwen snapshot not found: {snapshot}")
    from tunix.generate.sampler import Sampler  # type: ignore[import-untyped]
    model = model or load_qwen_single_device_model(snapshot)
    return cast(
        TextSampler,
        Sampler(model, tokenizer or load_qwen_tokenizer(snapshot), qwen_cache_config(cache_size)),
    )


class QwenTunixBackend:
    """Single-device Tunix Qwen implementation of the typed LLM backend."""

    def __init__(self, snapshot: Path, cache_size: int = 512, seed: int = 0) -> None:
        if cache_size < 2:
            raise ValueError("cache_size must reserve at least one prompt and one completion token")
        self._tokenizer = load_qwen_tokenizer(snapshot)
        self._model = load_qwen_single_device_model(snapshot)
        self._sampler = build_qwen_single_device_sampler(
            snapshot, cache_size, self._tokenizer, self._model
        )
        self._cache_size = cache_size
        self._seed = seed

    def complete(self, request: LlmRequest) -> LlmResponse:
        """Generate one raw completion and retain its latency/provenance."""
        if request.max_new_tokens >= self._cache_size:
            raise ValueError("max_new_tokens must be smaller than cache_size")
        started = perf_counter()
        chat_prompt = format_qwen_action_prompt(self._tokenizer, request.prompt.text)
        required_cache_size = qwen_required_cache_size(
            self._tokenizer, chat_prompt, request.max_new_tokens
        )
        if required_cache_size > self._cache_size:
            raise ValueError(
                "cache_size is too small for the padded Qwen prompt and completion: "
                f"requires at least {required_cache_size}, got {self._cache_size}"
            )
        output = self._sampler(
            chat_prompt,
            max_generation_steps=request.max_new_tokens,
            max_prompt_length=required_cache_size - request.max_new_tokens,
            temperature=request.temperature,
            seed=self._seed,
            return_logprobs=True,
        )
        raw_logprobs = output.logprobs
        token_logprobs = (
            tuple(float(value) for value in raw_logprobs[0]) if raw_logprobs else None
        )
        token_ids = tuple(int(token) for token in output.tokens[0])
        prompt_token_ids = tuple(
            int(token)
            for token in output.padded_prompt_tokens[0]
            if int(token) != self._tokenizer.pad_id()
        )
        return LlmResponse(
            raw_text=output.text[0],
            backend="tunix-single-device",
            model=QWEN_MODEL_ID,
            latency_ms=(perf_counter() - started) * 1_000,
            token_logprobs=token_logprobs,
            token_ids=token_ids,
            prompt_token_ids=prompt_token_ids,
        )

    def hidden_states(self, request: LlmRequest) -> jax.Array:
        """Return Qwen final hidden states ``[1, T, D]`` for one rendered chat prompt.

        This is a feature bridge only. A trainable critic/value head must be attached and
        checkpointed separately before PPO can consume these features as values.
        """
        chat_prompt = format_qwen_action_prompt(self._tokenizer, request.prompt.text)
        token_ids = qwen_chat_token_ids(self._tokenizer, chat_prompt)
        input_tokens = jax.numpy.asarray([token_ids], dtype=jax.numpy.int32)
        positions = jax.numpy.arange(input_tokens.shape[1], dtype=jax.numpy.int32)[None, :]
        hidden_states, _ = self._model(
            input_tokens, positions, cache=None, attention_mask=None, skip_lm_head=True
        )
        return hidden_states
