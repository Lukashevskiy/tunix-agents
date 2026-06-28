import jax
import jax.numpy as jnp
import numpy as np

from tunix_craftext.rollouts.reference import (
    collect_rollout,
    collect_rollout_scan,
    collect_rollout_scan_indexed,
)


def test_collect_rollout_is_time_major_and_preserves_terminal_flags() -> None:
    def policy(obs):
        return obs + 1, np.full(obs.shape, -0.5), np.full(obs.shape, 0.25)

    def step(state, action):
        next_state = state + action
        return (
            next_state,
            next_state,
            action.astype(float),
            state >= 3,
            np.zeros_like(state, dtype=bool),
        )

    _, _, batch = collect_rollout(np.array([0, 0]), np.array([0, 0]), 3, policy, step)

    assert batch.transitions.reward.shape == (3, 2)
    assert isinstance(batch.transitions.reward, jax.Array)
    assert isinstance(batch.bootstrap_value, jax.Array)
    np.testing.assert_array_equal(batch.transitions.action[:, 0], [1, 2, 4])
    np.testing.assert_array_equal(batch.transitions.terminated[:, 0], [False, False, True])
    np.testing.assert_array_equal(batch.bootstrap_value, [0.25, 0.25])


def test_jax_scan_collector_matches_reference_and_can_be_jitted() -> None:
    def policy(observation):
        return observation + 1, jnp.full(observation.shape, -0.5), jnp.full(observation.shape, 0.25)

    def step(state, action):
        next_state = state + action
        return (
            next_state,
            next_state,
            action.astype(jnp.float32),
            state >= 3,
            jnp.zeros_like(state, dtype=bool),
        )

    initial = jnp.array([0, 0])
    _, _, reference = collect_rollout(np.array([0, 0]), np.array([0, 0]), 3, policy, step)
    compiled = jax.jit(
        lambda state, observation: collect_rollout_scan(state, observation, 3, policy, step)
    )
    _, _, scanned = compiled(initial, initial)

    for reference_leaf, scanned_leaf in zip(
        jax.tree_util.tree_leaves(reference), jax.tree_util.tree_leaves(scanned)
    ):
        np.testing.assert_allclose(reference_leaf, scanned_leaf)


def test_indexed_scan_exposes_each_step_index_for_explicit_rng_selection() -> None:
    def policy(observation):
        return (
            jnp.zeros_like(observation, dtype=jnp.int32),
            jnp.zeros_like(observation),
            jnp.zeros_like(observation),
        )

    def step(state, action, index):
        next_state = state + index + 1
        reward = jnp.full_like(state, index + 1.0, dtype=jnp.float32)
        return (
            next_state,
            next_state,
            reward,
            jnp.zeros_like(state, dtype=bool),
            jnp.zeros_like(state, dtype=bool),
        )

    _, _, rollout = jax.jit(
        lambda state: collect_rollout_scan_indexed(state, state, 3, policy, step)
    )(jnp.zeros(2))

    np.testing.assert_array_equal(rollout.transitions.reward[:, 0], [1.0, 2.0, 3.0])
