import numpy as np
import pytest

from tunix_craftext.contracts import RolloutBatch, Transition


def test_rollout_contract_accepts_time_major_batch() -> None:
    transition = Transition(*(np.zeros((3, 2)) for _ in range(7)))
    RolloutBatch(transition, np.zeros(2)).validate()


def test_rollout_contract_rejects_non_time_major_rewards() -> None:
    transition = Transition(*(np.zeros(2) for _ in range(7)))
    with pytest.raises(ValueError, match="time-major"):
        RolloutBatch(transition, np.zeros(2)).validate()
