import numpy as np
import pytest
import jax
import jax.numpy as jnp

from tunix_craftext.contracts import RolloutBatch, Transition


def test_rollout_contract_accepts_time_major_batch() -> None:
    transition = Transition(*(np.zeros((3, 2)) for _ in range(7)))
    RolloutBatch(transition, np.zeros(2)).validate()


def test_rollout_contract_rejects_non_time_major_rewards() -> None:
    transition = Transition(*(np.zeros(2) for _ in range(7)))
    with pytest.raises(ValueError, match="time-major"):
        RolloutBatch(transition, np.zeros(2)).validate()


def test_contracts_are_registered_pytrees_and_done_can_be_jitted() -> None:
    transition = Transition(*(jnp.zeros((3, 2), dtype=bool) for _ in range(7)))
    batch = RolloutBatch(transition, jnp.zeros(2))

    assert len(jax.tree_util.tree_leaves(batch)) == 8
    np.testing.assert_array_equal(jax.jit(lambda value: value.done)(transition), np.zeros((3, 2), dtype=bool))


def test_rollout_contract_rejects_nested_observation_with_wrong_leading_axes() -> None:
    transition = Transition(
        observation={"tokens": jnp.zeros((3, 2)), "bad": jnp.zeros((3,))},
        action=jnp.zeros((3, 2)),
        reward=jnp.zeros((3, 2)),
        terminated=jnp.zeros((3, 2), dtype=bool),
        truncated=jnp.zeros((3, 2), dtype=bool),
        log_prob=jnp.zeros((3, 2)),
        value=jnp.zeros((3, 2)),
    )

    with pytest.raises(ValueError, match="observation leaf"):
        RolloutBatch(transition, jnp.zeros(2)).validate()
