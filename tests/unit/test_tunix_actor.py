"""Unit contracts for production-shaped Tunix Qwen/Gemma actors."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from tunix_craftext.llm import LlmRequest, LlmResponse
from tunix_craftext.prompts import ActionCatalog, RenderedPrompt
from tunix_craftext.tunix_actor import (
    CausalLmScoringModel,
    LinearValueHead,
    build_gemma_tunix_actor_from_components,
    causal_lm_actor_scores,
    causal_lm_actor_token_scores,
    causal_lm_critic_values,
    init_linear_value_head,
    merge_actor_critic_scores,
)

ROOT = Path(__file__).resolve().parents[2]


class _FakeCausalLm:
    """Small deterministic causal LM exposing the Tunix call subset."""

    vocab_size = 16
    hidden_dim = 3

    def __call__(
        self,
        input_tokens: jax.Array,
        positions: jax.Array,
        cache: object | None,
        attention_mask: jax.Array | None = None,
        *,
        skip_lm_head: bool = False,
    ) -> tuple[jax.Array, object | None]:
        del cache, attention_mask
        if input_tokens.shape != positions.shape:
            raise ValueError("positions must align with tokens")
        token_float = input_tokens.astype(jnp.float32)
        if skip_lm_head:
            hidden = jnp.stack(
                (
                    token_float,
                    positions.astype(jnp.float32),
                    jnp.ones_like(token_float),
                ),
                axis=-1,
            )
            return hidden, None

        vocab_positions = jnp.arange(self.vocab_size, dtype=jnp.float32)
        target = (input_tokens + positions + 1) % self.vocab_size
        logits = -jnp.abs(vocab_positions[None, None, :] - target[..., None].astype(jnp.float32))
        return logits, None


class _FakeBatchBackend:
    """Ordered backend double used by Tunix actor generation tests."""

    def complete(self, request: LlmRequest) -> LlmResponse:
        return self.complete_batch((request,))[0]

    def complete_batch(self, requests: Sequence[LlmRequest]) -> tuple[LlmResponse, ...]:
        return tuple(
            LlmResponse(
                raw_text=f"<action>LEFT</action> #{index}",
                backend="fake-tunix",
                model="fake-gemma",
                token_ids=(1, 2, index),
                token_logprobs=(-0.1, -0.2, -0.3),
                prompt_token_ids=(10, 11),
            )
            for index, _request in enumerate(requests)
        )


def _requests() -> tuple[LlmRequest, ...]:
    catalog = ActionCatalog(("LEFT", "RIGHT"))
    return (
        LlmRequest(RenderedPrompt("first prompt", catalog, "base"), max_new_tokens=3),
        LlmRequest(RenderedPrompt("second prompt", catalog, "base"), max_new_tokens=3),
    )


def test_init_linear_value_head_is_typed_and_finite() -> None:
    head = init_linear_value_head(jax.random.PRNGKey(0), hidden_dim=4)

    assert head.kernel.shape == (4,)
    assert head.bias.shape == ()
    assert bool(jnp.all(jnp.isfinite(head.kernel)))


def test_causal_lm_actor_scores_generated_tokens_and_masks_padding() -> None:
    head = LinearValueHead(
        kernel=jnp.array([0.5, 0.25, 1.0], dtype=jnp.float32),
        bias=jnp.asarray(-0.1, dtype=jnp.float32),
    )

    scores = causal_lm_actor_scores(
        _FakeCausalLm(),
        head,
        prompt_token_ids=jnp.array([[4, 5, 0], [7, 0, 0]], dtype=jnp.int32),
        prompt_token_mask=jnp.array([[True, True, False], [True, False, False]]),
        token_ids=jnp.array([[6, 8], [9, 0]], dtype=jnp.int32),
        token_mask=jnp.array([[True, True], [True, False]]),
    )

    assert scores.token_logprobs.shape == (2, 2)
    assert scores.values.shape == (2, 2)
    assert scores.entropy.shape == (2, 2)
    assert bool(jnp.all(jnp.isfinite(scores.token_logprobs)))
    assert bool(jnp.all(jnp.isfinite(scores.values)))
    assert float(scores.token_logprobs[1, 1]) == 0.0
    assert float(scores.values[1, 1]) == 0.0
    assert float(scores.entropy[1, 1]) == 0.0


def test_actor_and_critic_scores_are_separate_then_merge_at_objective_boundary() -> None:
    head = LinearValueHead(
        kernel=jnp.array([0.5, 0.25, 1.0], dtype=jnp.float32),
        bias=jnp.asarray(-0.1, dtype=jnp.float32),
    )
    prompt_token_ids = jnp.array([[4, 5, 0], [7, 0, 0]], dtype=jnp.int32)
    prompt_token_mask = jnp.array([[True, True, False], [True, False, False]])
    token_ids = jnp.array([[6, 8], [9, 0]], dtype=jnp.int32)
    token_mask = jnp.array([[True, True], [True, False]])

    actor_scores = causal_lm_actor_token_scores(
        _FakeCausalLm(),
        prompt_token_ids=prompt_token_ids,
        prompt_token_mask=prompt_token_mask,
        token_ids=token_ids,
        token_mask=token_mask,
    )
    critic_values = causal_lm_critic_values(
        _FakeCausalLm(),
        head,
        prompt_token_ids=prompt_token_ids,
        prompt_token_mask=prompt_token_mask,
        token_ids=token_ids,
        token_mask=token_mask,
    )
    merged = merge_actor_critic_scores(actor_scores, critic_values, token_ids)

    assert merged.token_logprobs.shape == token_ids.shape
    assert merged.values.shape == token_ids.shape
    assert float(merged.values[1, 1]) == 0.0


def test_gemma_tunix_actor_generates_and_scores_from_explicit_components() -> None:
    actor = build_gemma_tunix_actor_from_components(
        profile_path=ROOT / "configs/models/gemma3_270m_instruction.yaml",
        backend=_FakeBatchBackend(),
        model=_FakeCausalLm(),
        value_head=LinearValueHead(
            kernel=jnp.array([0.2, 0.1, 0.0], dtype=jnp.float32),
            bias=jnp.asarray(0.0, dtype=jnp.float32),
        ),
    )

    responses = actor.generate_batch(_requests())
    scores = actor.score_tokens(
        prompt_token_ids=jnp.array([[1, 2], [3, 4]], dtype=jnp.int32),
        prompt_token_mask=jnp.array([[True, True], [True, True]]),
        token_ids=jnp.array([[5, 6], [7, 8]], dtype=jnp.int32),
        token_mask=jnp.ones((2, 2), dtype=bool),
    )

    assert actor.profile.model_id == "google/gemma-3-270m-it"
    assert [response.raw_text for response in responses] == [
        "<action>LEFT</action> #0",
        "<action>LEFT</action> #1",
    ]
    assert scores.token_logprobs.shape == (2, 2)


def test_gemma_tunix_actor_exposes_separate_critic_role() -> None:
    actor = build_gemma_tunix_actor_from_components(
        profile_path=ROOT / "configs/models/gemma3_270m_instruction.yaml",
        backend=_FakeBatchBackend(),
        model=_FakeCausalLm(),
        value_head=LinearValueHead(
            kernel=jnp.array([0.2, 0.1, 0.0], dtype=jnp.float32),
            bias=jnp.asarray(0.0, dtype=jnp.float32),
        ),
    )
    actor_scores = actor.score_actor_tokens(
        prompt_token_ids=jnp.array([[1, 2]], dtype=jnp.int32),
        prompt_token_mask=jnp.array([[True, True]]),
        token_ids=jnp.array([[5, 6]], dtype=jnp.int32),
        token_mask=jnp.ones((1, 2), dtype=bool),
    )
    critic_values = actor.critic().score_values(
        prompt_token_ids=jnp.array([[1, 2]], dtype=jnp.int32),
        prompt_token_mask=jnp.array([[True, True]]),
        token_ids=jnp.array([[5, 6]], dtype=jnp.int32),
        token_mask=jnp.ones((1, 2), dtype=bool),
    )

    assert actor_scores.token_logprobs.shape == (1, 2)
    assert critic_values.values.shape == (1, 2)


def test_gemma_tunix_actor_rejects_qwen_profile() -> None:
    with pytest.raises(ValueError, match="gemma3"):
        build_gemma_tunix_actor_from_components(
            profile_path=ROOT / "configs/models/qwen25_05b_instruction.yaml",
            backend=_FakeBatchBackend(),
            model=_FakeCausalLm(),
            value_head=LinearValueHead(
                kernel=jnp.ones((3,), dtype=jnp.float32),
                bias=jnp.asarray(0.0, dtype=jnp.float32),
            ),
        )


def test_fake_model_satisfies_scoring_protocol() -> None:
    model: CausalLmScoringModel = _FakeCausalLm()
    logits, _ = model(
        jnp.ones((1, 2), dtype=jnp.int32),
        jnp.arange(2, dtype=jnp.int32)[None, :],
        cache=None,
    )

    assert logits.shape == (1, 2, _FakeCausalLm.vocab_size)
