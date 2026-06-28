import jax
import jax.numpy as jnp
import pytest

from tunix_craftext.rollouts.random_policy import (
    ActionSamplingError,
    sample_masked_actions,
    validate_action_mask,
)


def test_masked_random_policy_never_samples_invalid_actions() -> None:
    mask = jnp.array([[True, False, False], [False, True, True]])
    actions = sample_masked_actions(jax.random.split(jax.random.PRNGKey(7), 2), mask)

    assert actions.shape == (2,)
    assert bool(mask[0, actions[0]])
    assert bool(mask[1, actions[1]])


def test_masked_random_policy_is_safe_inside_jit() -> None:
    """Value checks remain at the host boundary, leaving sampling traceable."""
    mask = jnp.array([[True, False], [False, True]])
    keys = jax.random.split(jax.random.PRNGKey(3), 2)

    actions = jax.jit(sample_masked_actions)(keys, mask)

    assert actions.shape == (2,)
    assert bool(mask[0, actions[0]])
    assert bool(mask[1, actions[1]])


def test_masked_random_policy_rejects_empty_action_row() -> None:
    with pytest.raises(ActionSamplingError, match="at least"):
        validate_action_mask(jnp.array([[False, False]]))
