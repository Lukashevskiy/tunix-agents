"""Token learning-batch conversion tests with explicit fallback semantics."""

from __future__ import annotations

import numpy as np
import pytest

from tunix_craftext.replay import ReplayArtifact, ReplayStep
from tunix_craftext.text_trajectory import TextTrajectoryError, text_trajectory_from_replay


def test_replay_becomes_padded_token_batch_with_terminal_reward() -> None:
    artifact = ReplayArtifact(
        "config.yaml",
        "abc",
        "tunix",
        (
            ReplayStep(
                0,
                "p0",
                "c0",
                2,
                "DO",
                1.5,
                False,
                token_ids=(5, 6),
                token_logprobs=(-1.0, -2.0),
                prompt_token_ids=(101, 102, 103),
            ),
            ReplayStep(
                1,
                "p1",
                "c1",
                0,
                "NOOP",
                -0.5,
                True,
                fallback_used=True,
                token_ids=(7,),
                token_logprobs=(-3.0,),
                prompt_token_ids=(104,),
            ),
        ),
    )

    batch = text_trajectory_from_replay(artifact)

    assert batch.token_ids.shape == (2, 2)
    assert batch.prompt_token_ids.shape == (2, 3)
    np.testing.assert_array_equal(
        batch.prompt_token_mask, [[True, True, True], [True, False, False]]
    )
    np.testing.assert_array_equal(batch.token_mask, [[True, True], [True, False]])
    np.testing.assert_array_equal(batch.policy_mask, [[True, True], [False, False]])
    np.testing.assert_allclose(batch.rewards, [[0.0, 1.5], [-0.5, 0.0]])
    np.testing.assert_array_equal(batch.action_ids, [2, 0])
    np.testing.assert_array_equal(batch.terminated, [False, True])


def test_replay_without_aligned_token_provenance_is_rejected() -> None:
    artifact = ReplayArtifact(
        "config.yaml",
        "abc",
        "tunix",
        (
            ReplayStep(
                0,
                "p",
                "c",
                0,
                "NOOP",
                0.0,
                False,
                token_ids=(1,),
                token_logprobs=None,
                prompt_token_ids=(101,),
            ),
        ),
    )

    with pytest.raises(TextTrajectoryError, match="provenance"):
        text_trajectory_from_replay(artifact)
