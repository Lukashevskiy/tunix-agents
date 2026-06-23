"""JAX-native random policy for valid-action environment throughput baselines.

The module provides a simple random action sampler that respects explicit
action masks and can be used as a low-cost baseline for environment throughput
and scheduler testing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


class ActionSamplingError(ValueError):
    """Raised when an action mask cannot define a valid categorical distribution.

    Example:
        >>> raise ActionSamplingError("message")
    """


def sample_masked_actions(keys: jax.Array, action_mask: jax.Array) -> jax.Array:
    """Sample one valid discrete action per environment from boolean masks.

    :param keys: PRNG keys with shape ``[B, 2]``.
    :param action_mask: Valid-action mask with shape ``[B, A]`` where True marks allowed actions.
    :returns: Integer action ids with shape ``[B]``.
    :raises ActionSamplingError: If static mask/key axes disagree.

    Example:
        >>> actions = sample_masked_actions(keys, action_mask)
    """
    if keys.ndim != 2 or keys.shape[1] != 2 or action_mask.ndim != 2:
        raise ActionSamplingError("keys must be [B, 2] and action_mask must be [B, A]")
    if keys.shape[0] != action_mask.shape[0]:
        raise ActionSamplingError("keys and action_mask batch axes must agree")
    logits = jnp.where(action_mask, 0.0, -jnp.inf)
    return jax.vmap(jax.random.categorical)(keys, logits).astype(jnp.int32)


def validate_action_mask(action_mask: jax.Array) -> None:
    """Reject empty action rows at the non-jitted environment boundary.

    :param action_mask: Boolean array shaped ``[B, A]`` marking valid actions.
    :returns: None
    :raises ActionSamplingError: If any batch row contains zero valid actions.

    Example:
        >>> validate_action_mask(action_mask)
    """
    if action_mask.ndim != 2 or bool(jnp.any(jnp.logical_not(jnp.any(action_mask, axis=1)))):
        raise ActionSamplingError("every environment must have at least one valid action")
