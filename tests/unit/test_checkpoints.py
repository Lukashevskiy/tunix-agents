"""Contract tests for versioned PPO checkpoints."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tunix_craftext.checkpoints import CheckpointMetadata, restore_checkpoint, save_checkpoint
from tunix_craftext.learner import create_state, ppo_update


def test_checkpoint_round_trip_restores_optimizer_and_metadata(tmp_path) -> None:
    """A restored state produces the same next update as the original state."""
    state = create_state(jax.random.PRNGKey(0), observation_dim=3, actions=2)
    state, _ = ppo_update(
        state,
        observations=jnp.ones((4, 3)),
        actions=jnp.zeros(4, dtype=jnp.int32),
        old_log_prob=jnp.zeros(4),
        advantages=jnp.ones(4),
        returns=jnp.ones(4),
    )
    metadata = CheckpointMetadata(
        run_id="unit-checkpoint",
        config_digest="abc123",
        policy_kind="flax-actor-critic",
    )

    save_checkpoint(tmp_path / "checkpoint", state, metadata)
    restored, restored_metadata = restore_checkpoint(tmp_path / "checkpoint", state)

    assert restored_metadata == metadata
    assert int(restored.step) == int(state.step)
    expected, expected_metrics = ppo_update(
        state,
        jnp.ones((4, 3)),
        jnp.zeros(4, dtype=jnp.int32),
        jnp.zeros(4),
        jnp.ones(4),
        jnp.ones(4),
    )
    actual, actual_metrics = ppo_update(
        restored,
        jnp.ones((4, 3)),
        jnp.zeros(4, dtype=jnp.int32),
        jnp.zeros(4),
        jnp.ones(4),
        jnp.ones(4),
    )
    assert jax.tree_util.tree_all(
        jax.tree.map(lambda left, right: jnp.allclose(left, right), expected.params, actual.params)
    )
    assert jnp.allclose(expected_metrics["loss"], actual_metrics["loss"])


def test_checkpoint_rejects_unknown_schema(tmp_path) -> None:
    """A checkpoint made by a different schema never resumes silently."""
    state = create_state(jax.random.PRNGKey(1), observation_dim=3, actions=2)
    save_checkpoint(
        tmp_path / "checkpoint",
        state,
        CheckpointMetadata(run_id="schema", config_digest="digest", policy_kind="test"),
    )
    metadata_path = tmp_path / "checkpoint" / "tunix_craftext_metadata.json"
    metadata_path.write_text('{"schema":"unknown"}', encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        restore_checkpoint(tmp_path / "checkpoint", state)
