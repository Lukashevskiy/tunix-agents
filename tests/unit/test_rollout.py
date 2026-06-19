import numpy as np

from tunix_craftext.rollout import collect_rollout


def test_collect_rollout_is_time_major_and_preserves_terminal_flags() -> None:
    def policy(obs):
        return obs + 1, np.full(obs.shape, -0.5), np.full(obs.shape, 0.25)

    def step(state, action):
        next_state = state + action
        return next_state, next_state, action.astype(float), state >= 3, np.zeros_like(state, dtype=bool)

    _, _, batch = collect_rollout(np.array([0, 0]), np.array([0, 0]), 3, policy, step)

    assert batch.transitions.reward.shape == (3, 2)
    np.testing.assert_array_equal(batch.transitions.action[:, 0], [1, 2, 4])
    np.testing.assert_array_equal(batch.transitions.terminated[:, 0], [False, False, True])
    np.testing.assert_array_equal(batch.bootstrap_value, [0.25, 0.25])
