"""Production-shaped Tunix LLM actors for Qwen/Gemma backbones.

The actor boundary is intentionally split from concrete weight acquisition:
Qwen has a local-snapshot factory because the project already owns that path;
Gemma can be wired through explicit model/backend components until a no-download
checkpoint loader is accepted.  Both actors share the same causal-LM scoring
path: prompt + generated tokens go through the model, actor log-probabilities
are gathered autoregressively and a small value head produces critic values.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import jax
import jax.numpy as jnp

from .llm import BatchLlmBackend, LlmRequest, LlmResponse
from .llm_actor import LlmActorScores
from .model_profile import ModelProfile, load_model_profile
from .tensor_types import PromptTokenBatchBool, PromptTokenBatchInt, TokenBatchBool, TokenBatchInt
from .tunix_adapter import QwenTunixBackend, load_qwen_single_device_model


class CausalLmScoringModel(Protocol):
    """Tunix causal LM subset used for actor logprobs and value features."""

    def __call__(
        self,
        input_tokens: jax.Array,
        positions: jax.Array,
        cache: object | None,
        attention_mask: jax.Array | None = None,
        *,
        skip_lm_head: bool = False,
    ) -> tuple[jax.Array, object | None]:
        """Return logits or hidden states depending on ``skip_lm_head``."""
        ...


@dataclass(frozen=True)
class LinearValueHead:
    """Tiny trainable critic head attached to frozen/adapter-tuned LLM features."""

    kernel: jax.Array
    bias: jax.Array

    def __call__(self, hidden_states: jax.Array) -> jax.Array:
        """Project hidden states ``[B, T, D]`` to scalar values ``[B, T]``."""
        if hidden_states.shape[-1] != self.kernel.shape[0]:
            raise ValueError("hidden_states last axis must match value head kernel")
        return jnp.einsum("btd,d->bt", hidden_states, self.kernel) + self.bias


def init_linear_value_head(key: jax.Array, hidden_dim: int, scale: float = 0.01) -> LinearValueHead:
    """Initialize a small linear critic head for LLM token hidden states."""
    if hidden_dim <= 0 or scale <= 0:
        raise ValueError("hidden_dim and scale must be positive")
    return LinearValueHead(
        kernel=jax.random.normal(key, (hidden_dim,), dtype=jnp.float32) * scale,
        bias=jnp.asarray(0.0, dtype=jnp.float32),
    )


def causal_lm_actor_scores(
    model: CausalLmScoringModel,
    value_head: LinearValueHead,
    *,
    prompt_token_ids: PromptTokenBatchInt,
    prompt_token_mask: PromptTokenBatchBool,
    token_ids: TokenBatchInt,
    token_mask: TokenBatchBool,
) -> LlmActorScores:
    """Score generated tokens with a causal LM and value head.

    :param model: Tunix causal LM returning logits and hidden states.
    :param value_head: Critic head applied to hidden states at prediction positions.
    :param prompt_token_ids: Prompt ids shaped ``[B, P]``.
    :param prompt_token_mask: Prompt mask shaped ``[B, P]``.
    :param token_ids: Generated token ids shaped ``[B, T]``.
    :param token_mask: Generated token mask shaped ``[B, T]``.
    :returns: Token logprobs, values, entropy and mask shaped ``[B, T]``.
    """
    _validate_token_axes(prompt_token_ids, prompt_token_mask, token_ids, token_mask)
    input_tokens = jnp.concatenate((prompt_token_ids, token_ids), axis=1)
    positions = jnp.broadcast_to(
        jnp.arange(input_tokens.shape[1], dtype=jnp.int32), input_tokens.shape
    )
    logits, _ = model(input_tokens, positions, cache=None, attention_mask=None, skip_lm_head=False)
    hidden_states, _ = model(
        input_tokens, positions, cache=None, attention_mask=None, skip_lm_head=True
    )
    if logits.ndim != 3 or logits.shape[:2] != input_tokens.shape:
        raise ValueError("model logits must have shape [B, P + T, V]")
    if hidden_states.ndim != 3 or hidden_states.shape[:2] != input_tokens.shape:
        raise ValueError("hidden states must have shape [B, P + T, D]")

    batch_size, token_length = token_ids.shape
    prompt_lengths = jnp.sum(prompt_token_mask.astype(jnp.int32), axis=1)
    token_offsets = jnp.arange(token_length, dtype=jnp.int32)[None, :]
    score_positions = jnp.maximum(prompt_lengths[:, None] + token_offsets - 1, 0)
    batch_positions = jnp.arange(batch_size, dtype=jnp.int32)[:, None]
    token_logits = logits[batch_positions, score_positions]
    token_hidden = hidden_states[batch_positions, score_positions]

    log_probs = jax.nn.log_softmax(token_logits, axis=-1)
    selected_logprobs = jnp.take_along_axis(log_probs, token_ids[..., None], axis=-1).squeeze(-1)
    probabilities = jax.nn.softmax(token_logits, axis=-1)
    entropy = -jnp.sum(probabilities * log_probs, axis=-1)
    values = value_head(token_hidden)
    scores = LlmActorScores(
        token_logprobs=jnp.where(token_mask, selected_logprobs, 0.0),
        values=jnp.where(token_mask, values, 0.0),
        entropy=jnp.where(token_mask, entropy, 0.0),
        token_mask=token_mask,
    )
    scores.validate(token_ids)
    return scores


@dataclass(frozen=True)
class TunixCausalLmActor:
    """Shared Tunix-backed LLM actor implementation."""

    profile: ModelProfile
    backend: BatchLlmBackend
    model: CausalLmScoringModel
    value_head: LinearValueHead

    def generate_batch(self, requests: Sequence[LlmRequest]) -> tuple[LlmResponse, ...]:
        """Generate ordered completions through the configured backend."""
        return self.backend.complete_batch(requests)

    def score_tokens(
        self,
        *,
        prompt_token_ids: PromptTokenBatchInt,
        prompt_token_mask: PromptTokenBatchBool,
        token_ids: TokenBatchInt,
        token_mask: TokenBatchBool,
    ) -> LlmActorScores:
        """Recompute token logprobs, values and entropy through the causal LM."""
        return causal_lm_actor_scores(
            self.model,
            self.value_head,
            prompt_token_ids=prompt_token_ids,
            prompt_token_mask=prompt_token_mask,
            token_ids=token_ids,
            token_mask=token_mask,
        )


@dataclass(frozen=True)
class QwenTunixActor(TunixCausalLmActor):
    """Qwen actor using the existing local-snapshot Tunix backend."""


@dataclass(frozen=True)
class GemmaTunixActor(TunixCausalLmActor):
    """Gemma actor wired from explicit Tunix model/backend components."""


def build_qwen_tunix_actor(
    *,
    profile_path: Path,
    snapshot: Path,
    cache_size: int,
    value_head: LinearValueHead,
    seed: int = 0,
) -> QwenTunixActor:
    """Build a Qwen LLM actor from explicit local weights without downloads."""
    profile = load_model_profile(profile_path)
    if profile.architecture != "qwen2":
        raise ValueError("QwenTunixActor requires a qwen2 model profile")
    backend = QwenTunixBackend(snapshot, cache_size=cache_size, seed=seed)
    model = cast(CausalLmScoringModel, load_qwen_single_device_model(snapshot))
    return QwenTunixActor(profile, backend, model, value_head)


def build_gemma_tunix_actor_from_components(
    *,
    profile_path: Path,
    backend: BatchLlmBackend,
    model: CausalLmScoringModel,
    value_head: LinearValueHead,
) -> GemmaTunixActor:
    """Build a Gemma actor from already loaded Tunix components.

    The project intentionally does not download Gemma weights implicitly.  A
    future loader may call Tunix Gemma checkpoint/safetensors helpers once the
    exact source, licence acknowledgement and local snapshot path are recorded.
    """
    profile = load_model_profile(profile_path)
    if profile.architecture != "gemma3":
        raise ValueError("GemmaTunixActor requires a gemma3 model profile")
    return GemmaTunixActor(profile, backend, model, value_head)


def _validate_token_axes(
    prompt_token_ids: jax.Array,
    prompt_token_mask: jax.Array,
    token_ids: jax.Array,
    token_mask: jax.Array,
) -> None:
    if prompt_token_ids.ndim != 2 or prompt_token_mask.shape != prompt_token_ids.shape:
        raise ValueError("prompt_token_ids and prompt_token_mask must have shape [B, P]")
    if token_ids.ndim != 2 or token_mask.shape != token_ids.shape:
        raise ValueError("token_ids and token_mask must have shape [B, T]")
    if prompt_token_ids.shape[0] != token_ids.shape[0]:
        raise ValueError("prompt and token batches must share the same B axis")
