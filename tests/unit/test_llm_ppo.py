"""Tests for PPO evaluation from real LLM actor/critic scores."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tunix_craftext.llm_actor import LlmActorScores
from tunix_craftext.llm_ppo import (
    evaluate_llm_actor_critic_ppo,
    evaluate_separate_llm_actor_critic_ppo,
)
from tunix_craftext.text_trajectory import TextTrajectoryBatch
from tunix_craftext.tunix_actor import LlmActorTokenScores, LlmCriticValues


def _batch() -> TextTrajectoryBatch:
    return TextTrajectoryBatch(
        token_ids=jnp.array([[1, 2, 0], [3, 4, 5]], dtype=jnp.int32),
        prompt_token_ids=jnp.array([[10, 11], [12, 0]], dtype=jnp.int32),
        prompt_token_mask=jnp.array([[True, True], [True, False]]),
        old_logprobs=jnp.array([[-0.3, -0.2, 0.0], [-0.5, -0.4, -0.1]], dtype=jnp.float32),
        token_mask=jnp.array([[True, True, False], [True, True, True]]),
        policy_mask=jnp.array([[True, True, False], [True, True, True]]),
        rewards=jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, -0.2]], dtype=jnp.float32),
        action_ids=jnp.array([1, 2], dtype=jnp.int32),
        terminated=jnp.array([False, True]),
        fallback_used=jnp.array([False, False]),
    )


def _scores() -> LlmActorScores:
    return LlmActorScores(
        token_logprobs=jnp.array([[-0.25, -0.22, 0.0], [-0.55, -0.35, -0.2]], dtype=jnp.float32),
        values=jnp.array([[0.1, 0.2, 0.0], [0.0, -0.1, -0.2]], dtype=jnp.float32),
        entropy=jnp.array([[0.8, 0.7, 0.0], [0.9, 0.6, 0.5]], dtype=jnp.float32),
        token_mask=jnp.array([[True, True, False], [True, True, True]]),
    )


def test_evaluate_llm_actor_critic_ppo_uses_real_scores() -> None:
    evaluation = evaluate_llm_actor_critic_ppo(_batch(), _scores(), gamma=0.9)

    assert evaluation.returns.shape == (2, 3)
    assert evaluation.advantages.shape == (2, 3)
    assert evaluation.loss.shape == ()
    assert float(evaluation.metrics["learned_tokens"]) == 5.0
    assert bool(jnp.isfinite(evaluation.loss))


def test_evaluate_llm_actor_critic_ppo_respects_custom_learning_mask() -> None:
    mask = jnp.array([[True, False, False], [False, True, False]])

    evaluation = evaluate_llm_actor_critic_ppo(_batch(), _scores(), learning_mask=mask)

    assert float(evaluation.metrics["learned_tokens"]) == 2.0


def test_evaluate_separate_llm_actor_critic_ppo_merges_only_at_loss_boundary() -> None:
    scores = _scores()
    actor_scores = LlmActorTokenScores(
        token_logprobs=scores.token_logprobs,
        entropy=scores.entropy,
        token_mask=scores.token_mask,
    )
    critic_values = LlmCriticValues(values=scores.values, token_mask=scores.token_mask)

    evaluation = evaluate_separate_llm_actor_critic_ppo(
        _batch(), actor_scores, critic_values, gamma=0.9
    )

    assert evaluation.returns.shape == (2, 3)
    assert bool(jnp.isfinite(evaluation.loss))


def test_evaluate_llm_actor_critic_ppo_rejects_non_finite_scores() -> None:
    scores = LlmActorScores(
        token_logprobs=jnp.array([[jnp.nan, -0.22, 0.0], [-0.55, -0.35, -0.2]]),
        values=_scores().values,
        entropy=_scores().entropy,
        token_mask=_scores().token_mask,
    )

    with pytest.raises(ValueError, match="token_logprobs"):
        evaluate_llm_actor_critic_ppo(_batch(), scores)
