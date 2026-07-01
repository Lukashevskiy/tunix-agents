"""Tests for external GRPO evaluation from real actor score contracts."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tunix_craftext.artifacts.replay import ReplayArtifact, ReplayStep
from tunix_craftext.models.tunix_actor import LlmActorTokenScores
from tunix_craftext.research.llm_grpo import evaluate_external_llm_actor_grpo
from tunix_craftext.training.external_grpo import (
    external_grpo_batch_from_replays,
    token_batch_from_external_grpo,
)


def _token_batch():
    return token_batch_from_external_grpo(
        external_grpo_batch_from_replays(
            goal="collect wood",
            group_prefix="wood",
            group_size=2,
            replays=(_replay(1.0), _replay(3.0)),
        )
    )


def _replay(total_reward: float) -> ReplayArtifact:
    return ReplayArtifact(
        config_path="configs/env/text/qwen_craftext.yaml",
        commit="abc123",
        backend="vllm-offload",
        steps=(
            ReplayStep(
                index=0,
                prompt="goal",
                raw_completion="<action>NOOP</action>",
                action_id=0,
                action_label="NOOP",
                reward=total_reward,
                terminated=False,
                token_ids=(11, 12),
                token_logprobs=(-0.1, -0.2),
                prompt_token_ids=(1, 2, 3),
            ),
        ),
    )


def test_evaluate_external_llm_actor_grpo_uses_group_advantages() -> None:
    batch = _token_batch()
    scores = LlmActorTokenScores(
        token_logprobs=batch.old_logprobs + jnp.asarray([[0.1, 0.1], [-0.1, -0.1]]),
        entropy=jnp.full(batch.token_ids.shape, 0.5, dtype=jnp.float32),
        token_mask=batch.token_mask,
    )

    evaluation = evaluate_external_llm_actor_grpo(
        batch,
        scores,
        entropy_coefficient=0.01,
    )

    assert evaluation.loss.shape == ()
    assert bool(jnp.isfinite(evaluation.loss))
    assert float(evaluation.metrics["learned_tokens"]) == 4.0
    assert float(evaluation.metrics["mean_sample_reward"]) == 2.0


def test_evaluate_external_llm_actor_grpo_rejects_missing_actor_tokens() -> None:
    batch = _token_batch()
    scores = LlmActorTokenScores(
        token_logprobs=batch.old_logprobs,
        entropy=jnp.ones(batch.token_ids.shape),
        token_mask=jnp.array([[True, True], [True, False]]),
    )

    with pytest.raises(ValueError, match="learning_mask"):
        evaluate_external_llm_actor_grpo(batch, scores)
