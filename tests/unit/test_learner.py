import jax
import jax.numpy as jnp
import pytest

from tunix_craftext.artifacts.replay import ReplayArtifact, ReplayStep
from tunix_craftext.artifacts.text_trajectory import TextTrajectoryBatch
from tunix_craftext.research.learner import (
    create_state,
    create_token_state,
    external_grpo_actor_outputs,
    external_grpo_update,
    full_token_ppo_update,
    ppo_update,
    token_actor_critic_outputs,
    token_ppo_update,
)
from tunix_craftext.training.external_grpo import (
    external_grpo_batch_from_replays,
    token_batch_from_external_grpo,
)


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
        invalid_action=jnp.array([False, False]),
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


def test_full_token_ppo_update_learns_from_all_generated_tokens_including_fallback() -> None:
    state = create_token_state(jax.random.PRNGKey(3), token_bucket_count=16, hidden=8)
    batch = _token_batch(fallback=True)

    updated, metrics = full_token_ppo_update(state, batch)

    assert bool(jnp.isfinite(metrics["loss"]))
    assert float(metrics["learned_tokens"]) == float(jnp.sum(batch.token_mask))
    assert int(metrics["learning_mode"]) == 1
    assert not bool(
        jnp.allclose(
            state.params["Embed_0"]["embedding"],
            updated.params["Embed_0"]["embedding"],
        )
    )


def test_policy_token_ppo_reports_only_policy_masked_tokens() -> None:
    state = create_token_state(jax.random.PRNGKey(4), token_bucket_count=16, hidden=8)
    batch = _token_batch()

    _, metrics = token_ppo_update(state, batch)

    assert float(metrics["learned_tokens"]) == float(jnp.sum(batch.policy_mask))
    assert int(metrics["learning_mode"]) == 0


def _external_grpo_token_batch():
    replays = (
        _grpo_replay(1.0, token_ids=(3, 4), logprobs=(-1.0, -1.1)),
        _grpo_replay(3.0, token_ids=(5, 6), logprobs=(-1.2, -1.3)),
    )
    return token_batch_from_external_grpo(
        external_grpo_batch_from_replays(
            goal="collect wood",
            group_prefix="wood",
            replays=replays,
            group_size=2,
        )
    )


def _grpo_replay(
    total_reward: float, *, token_ids: tuple[int, ...], logprobs: tuple[float, ...]
) -> ReplayArtifact:
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
                token_ids=token_ids,
                token_logprobs=logprobs,
                prompt_token_ids=(1, 2, 3),
            ),
        ),
    )


def test_external_grpo_actor_outputs_match_external_token_batch_axes() -> None:
    state = create_token_state(jax.random.PRNGKey(5), token_bucket_count=16, hidden=8)
    batch = _external_grpo_token_batch()

    scores = external_grpo_actor_outputs(state, batch)

    assert scores.token_logprobs.shape == batch.token_ids.shape
    assert scores.entropy.shape == batch.token_ids.shape
    assert scores.token_mask.tolist() == batch.token_mask.tolist()
    assert bool(jnp.all(jnp.isfinite(scores.token_logprobs[batch.token_mask])))


def test_external_grpo_update_recomputes_actor_and_changes_parameters() -> None:
    state = create_token_state(jax.random.PRNGKey(6), token_bucket_count=16, hidden=8)
    batch = _external_grpo_token_batch()

    updated, metrics = external_grpo_update(state, batch, entropy_coefficient=0.01)

    assert bool(jnp.isfinite(metrics["loss"]))
    assert bool(jnp.isfinite(metrics["policy_loss"]))
    assert float(metrics["learned_tokens"]) == float(jnp.sum(batch.token_mask))
    assert float(metrics["mean_sample_reward"]) == 2.0
    assert not bool(
        jnp.allclose(
            state.params["Embed_0"]["embedding"],
            updated.params["Embed_0"]["embedding"],
        )
    )
