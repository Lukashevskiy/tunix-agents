"""Backbone-neutral LLM actor contracts for token-level RL.

This module defines the boundary we want production Qwen/Gemma actors to
implement: generation for environment interaction and scoring for PPO/GRPO
updates.  The deterministic actor is intentionally tiny and exists only for
contract tests; real backbones should implement the same protocol through Tunix
model/sampler adapters.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import jax.numpy as jnp

from ..core.tensor_types import (
    PromptTokenBatchBool,
    PromptTokenBatchInt,
    TokenBatchBool,
    TokenBatchFloat,
    TokenBatchInt,
)
from .llm import LlmRequest, LlmResponse
from .profile import ModelProfile


@dataclass(frozen=True)
class LlmActorScores:
    """Token-level actor scores shaped for PPO/GRPO updates."""

    token_logprobs: TokenBatchFloat
    values: TokenBatchFloat
    entropy: TokenBatchFloat
    token_mask: TokenBatchBool

    def validate(self, token_ids: TokenBatchInt) -> None:
        """Validate that score arrays align with generated token ids.

        :param token_ids: Generated token ids shaped ``[B, T]``.
        :raises ValueError: If any score tensor has a mismatched shape.
        """
        expected = tuple(token_ids.shape)
        for name in ("token_logprobs", "values", "entropy", "token_mask"):
            if tuple(getattr(self, name).shape) != expected:
                raise ValueError(f"{name} must have shape {expected}")


class LlmActor(Protocol):
    """Trainable or frozen LLM actor boundary used by rollout and update code."""

    @property
    def profile(self) -> ModelProfile:
        """Backbone profile that owns model identity and resource provenance."""
        ...

    def generate_batch(self, requests: Sequence[LlmRequest]) -> tuple[LlmResponse, ...]:
        """Generate completions for an ordered prompt batch."""
        ...

    def score_tokens(
        self,
        *,
        prompt_token_ids: PromptTokenBatchInt,
        prompt_token_mask: PromptTokenBatchBool,
        token_ids: TokenBatchInt,
        token_mask: TokenBatchBool,
    ) -> LlmActorScores:
        """Recompute token log-probabilities, critic values and entropy."""
        ...


@dataclass(frozen=True)
class DeterministicLlmActor:
    """Small deterministic LLM-actor test double with production-shaped outputs."""

    profile: ModelProfile
    action_text: str = "<action>NOOP</action>"
    token_bucket_count: int = 128

    def generate_batch(self, requests: Sequence[LlmRequest]) -> tuple[LlmResponse, ...]:
        """Return deterministic completions with token provenance for every request."""
        if not requests:
            raise ValueError("requests must be non-empty")
        token_ids = tuple(ord(char) % self.token_bucket_count for char in self.action_text)
        token_logprobs = tuple(-0.1 for _ in token_ids)
        return tuple(
            LlmResponse(
                raw_text=self.action_text,
                backend="deterministic-llm-actor",
                model=self.profile.model_id,
                latency_ms=0.0,
                token_logprobs=token_logprobs,
                token_ids=token_ids,
                prompt_token_ids=tuple(range(1, 1 + len(request.prompt.text.split()))),
            )
            for request in requests
        )

    def score_tokens(
        self,
        *,
        prompt_token_ids: PromptTokenBatchInt,
        prompt_token_mask: PromptTokenBatchBool,
        token_ids: TokenBatchInt,
        token_mask: TokenBatchBool,
    ) -> LlmActorScores:
        """Produce deterministic finite scores with the same axes as ``token_ids``."""
        if prompt_token_ids.ndim != 2 or prompt_token_mask.shape != prompt_token_ids.shape:
            raise ValueError("prompt_token_ids and prompt_token_mask must have shape [B, P]")
        if token_ids.ndim != 2 or token_mask.shape != token_ids.shape:
            raise ValueError("token_ids and token_mask must have shape [B, T]")
        if prompt_token_ids.shape[0] != token_ids.shape[0]:
            raise ValueError("prompt and token batches must have the same B axis")
        prompt_scale = jnp.maximum(
            jnp.sum(prompt_token_mask.astype(jnp.float32), axis=1, keepdims=True), 1.0
        )
        bucketed = jnp.mod(token_ids, self.token_bucket_count).astype(jnp.float32)
        logprobs = -jnp.log1p(bucketed) / prompt_scale
        values = jnp.tanh(bucketed / float(self.token_bucket_count))
        entropy = jnp.where(token_mask, jnp.full_like(values, 0.5), 0.0)
        scores = LlmActorScores(logprobs, values, entropy, token_mask)
        scores.validate(token_ids)
        return scores
