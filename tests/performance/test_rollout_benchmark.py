import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tunix_craftext.rollouts.reference import collect_rollout, collect_rollout_scan


@pytest.mark.performance
def test_reference_rollout_benchmark(benchmark) -> None:
    def policy(obs):
        return obs, np.zeros_like(obs), np.zeros_like(obs)

    def step(state, action):
        return state + 1, state, np.ones_like(state), state > 100, np.zeros_like(state, dtype=bool)

    benchmark(collect_rollout, np.zeros(256), np.zeros(256), 128, policy, step)


@pytest.mark.performance
def test_jax_scan_rollout_steady_state_benchmark(benchmark) -> None:
    def policy(obs):
        return obs, jnp.zeros_like(obs), jnp.zeros_like(obs)

    def step(state, action):
        return (
            state + 1,
            state,
            jnp.ones_like(state),
            state > 100,
            jnp.zeros_like(state, dtype=bool),
        )

    compiled = jax.jit(
        lambda state, observation: collect_rollout_scan(state, observation, 128, policy, step)
    )
    initial = jnp.zeros(256)
    compiled(initial, initial)[
        2
    ].bootstrap_value.block_until_ready()  # Deliberate compile/warmup outside benchmark.
    benchmark(lambda: compiled(initial, initial)[2].bootstrap_value.block_until_ready())
