import numpy as np
import pytest

from tunix_craftext.rollout import collect_rollout


@pytest.mark.performance
def test_reference_rollout_benchmark(benchmark) -> None:
    def policy(obs):
        return obs, np.zeros_like(obs), np.zeros_like(obs)

    def step(state, action):
        return state + 1, state, np.ones_like(state), state > 100, np.zeros_like(state, dtype=bool)

    benchmark(collect_rollout, np.zeros(256), np.zeros(256), 128, policy, step)
