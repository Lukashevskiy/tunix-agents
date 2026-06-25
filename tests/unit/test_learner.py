import jax
import jax.numpy as jnp
import pytest

from tunix_craftext.learner import (
    create_state,
    create_token_state,
    ppo_update,
    token_actor_critic_outputs,
    token_ppo_update,
)
from tunix_craftext.text_trajectory import TextTrajectoryBatch


def test_ppo_update_is_finite_and_changes_parameters() -> None:
    state = create_state(jax.random.PRNGKey(0), 3, 2)
    updated, metrics = ppo_update(
        state,
        jnp.ones((4, 3)),
        jnp.zeros(4, dtype=jnp.int32),
        jnp.zeros(4),
        jnp.ones(4),
        jnp.ones(4),
    )
    assert bool(jnp.isfinite(metrics["loss"]))
    assert not bool(
        jnp.allclose(state.params["Dense_0"]["kernel"], updated.params["Dense_0"]["kernel"])
    )


def _token_batch(*, fallback: bool = False) -> TextTrajectoryBatch:
    token_mask = jnp.array([[True, True, False], [True, True, True]])
    fallback_used = jnp.array([fallback, fallback])
    return TextTrajectoryBatch(
        token_ids=jnp.array([[3, 4, 0], [5, 6, 7]], dtype=jnp.int32),
        prompt_token_ids=jnp.array([[11, 12], [13, 0]], dtype=jnp.int32),
        prompt_token_mask=jnp.array([[True, True], [True, False]]),
        old_logprobs=jnp.full((2, 3), -1.0),
        token_mask=token_mask,
        policy_mask=jnp.logical_and(token_mask, ~fallback_used[:, None]),
        rewards=jnp.array([[0.0, 1.0, 0.0], [0.0, 0.0, -0.5]], dtype=jnp.float32),
        action_ids=jnp.array([1, 2], dtype=jnp.int32),
        terminated=jnp.array([False, True]),
        fallback_used=fallback_used,
    )


def test_token_actor_critic_outputs_match_text_batch_axes() -> None:
    state = create_token_state(jax.random.PRNGKey(0), token_bucket_count=16, hidden=8)
    batch = _token_batch()

    logprobs, values, entropy = token_actor_critic_outputs(state, batch)

    assert logprobs.shape == batch.token_ids.shape
    assert values.shape == batch.token_ids.shape
    assert entropy.shape == batch.token_ids.shape
    assert bool(jnp.all(jnp.isfinite(logprobs)))


def test_token_ppo_update_recomputes_actor_critic_and_changes_parameters() -> None:
    state = create_token_state(jax.random.PRNGKey(1), token_bucket_count=16, hidden=8)
    batch = _token_batch()

    updated, metrics = token_ppo_update(state, batch)

    assert bool(jnp.isfinite(metrics["loss"]))
    assert bool(jnp.isfinite(metrics["value_loss"]))
    assert not bool(
        jnp.allclose(
            state.params["Embed_0"]["embedding"],
            updated.params["Embed_0"]["embedding"],
        )
    )


def test_token_ppo_update_rejects_fallback_only_policy_batch() -> None:
    state = create_token_state(jax.random.PRNGKey(2), token_bucket_count=16, hidden=8)

    with pytest.raises(ValueError, match="token_mask"):
        token_ppo_update(state, _token_batch(fallback=True))
