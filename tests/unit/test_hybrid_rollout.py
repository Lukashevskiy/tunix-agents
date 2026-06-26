"""Hybrid rollout contracts for PPO-ready agentic LLM data."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tunix_craftext.hybrid_rollout import (
    HybridPpoStep,
    compute_masked_step_token_ppo_loss,
    hybrid_trajectory_from_steps,
)


def _step(*, step_mask: jnp.ndarray | None = None) -> HybridPpoStep:
    """Build one valid two-row hybrid PPO step."""
    return HybridPpoStep(
        action_ids=jnp.asarray([1, 0], dtype=jnp.int32),
        prompt_tokens=jnp.asarray([[1, 2, 0], [3, 4, 5]], dtype=jnp.int32),
        prompt_token_mask=jnp.asarray([[True, True, False], [True, True, True]]),
        generation_tokens=jnp.asarray([[10, 11], [20, 0]], dtype=jnp.int32),
        generation_token_mask=jnp.asarray([[True, True], [True, False]]),
        actor_log_probs=jnp.asarray([[-0.2, -0.3], [-0.7, 0.0]], dtype=jnp.float32),
        values=jnp.asarray([0.5, -0.25], dtype=jnp.float32),
        step_mask=jnp.asarray([True, False]) if step_mask is None else step_mask,
        action_mask=jnp.asarray([[True, True], [True, False]]),
    )


def test_hybrid_ppo_step_accepts_actor_critic_token_contract() -> None:
    step = _step()

    step.validate()

    assert step.action_ids.shape == (2,)
    assert step.generation_tokens.shape == step.actor_log_probs.shape
    assert step.step_mask.tolist() == [True, False]


def test_hybrid_ppo_step_rejects_mismatched_token_logprobs() -> None:
    step = _step()
    broken = HybridPpoStep(
        action_ids=step.action_ids,
        prompt_tokens=step.prompt_tokens,
        prompt_token_mask=step.prompt_token_mask,
        generation_tokens=step.generation_tokens,
        generation_token_mask=step.generation_token_mask,
        actor_log_probs=jnp.zeros((2, 3), dtype=jnp.float32),
        values=step.values,
        step_mask=step.step_mask,
    )

    with pytest.raises(ValueError, match="actor_log_probs"):
        broken.validate()


def test_hybrid_trajectory_stacks_step_masks_time_major() -> None:
    trajectory = hybrid_trajectory_from_steps(
        (
            _step(step_mask=jnp.asarray([True, True])),
            _step(step_mask=jnp.asarray([True, False])),
        )
    )

    assert trajectory.step_masks.shape == (2, 2)
    assert trajectory.step_masks.tolist() == [[True, True], [True, False]]


def test_hybrid_trajectory_rejects_inconsistent_batch_size() -> None:
    first = _step(step_mask=jnp.asarray([True, True]))
    second = HybridPpoStep(
        action_ids=jnp.asarray([1], dtype=jnp.int32),
        prompt_tokens=jnp.asarray([[1, 2]], dtype=jnp.int32),
        prompt_token_mask=jnp.asarray([[True, True]]),
        generation_tokens=jnp.asarray([[10]], dtype=jnp.int32),
        generation_token_mask=jnp.asarray([[True]]),
        actor_log_probs=jnp.asarray([[-0.2]], dtype=jnp.float32),
        values=jnp.asarray([0.0], dtype=jnp.float32),
        step_mask=jnp.asarray([True]),
    )

    with pytest.raises(ValueError, match="step_mask"):
        hybrid_trajectory_from_steps((first, second))


def test_masked_step_token_ppo_loss_ignores_dead_episode_rows() -> None:
    new_log_probs = jnp.asarray([[-0.2, -0.2], [-100.0, -100.0]], dtype=jnp.float32)
    old_log_probs = jnp.asarray([[-0.2, -0.2], [0.0, 0.0]], dtype=jnp.float32)
    token_mask = jnp.asarray([[True, True], [True, True]])
    advantages = jnp.asarray([1.0, 1000.0], dtype=jnp.float32)
    step_mask = jnp.asarray([True, False])

    loss = compute_masked_step_token_ppo_loss(
        new_log_probs,
        old_log_probs,
        token_mask,
        advantages,
        step_mask,
        clip_epsilon=0.2,
    )

    assert loss == pytest.approx(-1.0)


def test_masked_step_token_ppo_loss_ignores_generated_padding_tokens() -> None:
    loss = compute_masked_step_token_ppo_loss(
        jnp.asarray([[-0.2, -100.0]], dtype=jnp.float32),
        jnp.asarray([[-0.2, 100.0]], dtype=jnp.float32),
        jnp.asarray([[True, False]]),
        jnp.asarray([2.0], dtype=jnp.float32),
        jnp.asarray([True]),
    )

    assert loss == pytest.approx(-2.0)


def test_masked_step_token_ppo_loss_requires_valid_tokens() -> None:
    with pytest.raises(ValueError, match="valid generated token"):
        compute_masked_step_token_ppo_loss(
            jnp.zeros((1, 2)),
            jnp.zeros((1, 2)),
            jnp.asarray([[False, False]]),
            jnp.ones((1,)),
            jnp.asarray([True]),
        )
