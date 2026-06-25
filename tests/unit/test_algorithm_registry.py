"""TDD contracts for interchangeable pure RL objectives."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from tunix_craftext.research.algorithm_registry import PpoLossBatch, get_algorithm


def test_ppo_registry_loss_is_jittable_and_returns_metric_pytree() -> None:
    """PPO entry is a pure function usable by future generic learners."""
    spec = get_algorithm("ppo")
    batch = PpoLossBatch(
        jnp.zeros(2),
        jnp.zeros(2),
        jnp.array([1.0, -1.0]),
        jnp.zeros(2),
        jnp.zeros(2),
        jnp.zeros(2),
        jnp.ones(2),
    )
    output = jax.jit(spec.loss)(batch)
    assert bool(jnp.isfinite(output.loss))
    assert set(output.metrics) >= {"loss", "policy_loss", "value_loss", "entropy", "approx_kl"}


def test_registry_rejects_unknown_algorithm() -> None:
    """Experiment configs never fall back silently to a different objective."""
    with pytest.raises(ValueError, match="unknown algorithm"):
        get_algorithm("gspo")
