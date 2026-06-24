"""Regression tests for project-level jaxtyping public contracts."""

from typing import get_type_hints

from tunix_craftext.contracts import RolloutBatch, Transition
from tunix_craftext.random_policy import sample_masked_actions
from tunix_craftext.tensor_types import (
    ActionMask,
    BatchFloat,
    BatchLegacyKey,
    TimeBatchBool,
    TimeBatchFloat,
    TokenBatchBool,
    TokenBatchFloat,
)
from tunix_craftext.text_trajectory import TextTrajectoryBatch


def test_rollout_and_token_boundaries_expose_jaxtyping_axis_aliases() -> None:
    """Prevent regressions back to unshaped ``jax.Array`` public contracts."""
    transition_hints = get_type_hints(Transition)
    rollout_hints = get_type_hints(RolloutBatch)
    token_hints = get_type_hints(TextTrajectoryBatch)

    assert transition_hints["reward"] == TimeBatchFloat
    assert transition_hints["terminated"] == TimeBatchBool
    assert transition_hints["log_prob"] == TimeBatchFloat
    assert rollout_hints["bootstrap_value"] == BatchFloat
    assert token_hints["old_logprobs"] == TokenBatchFloat
    assert token_hints["policy_mask"] == TokenBatchBool


def test_masked_sampler_signature_keeps_key_mask_and_action_axes_explicit() -> None:
    """The throughput baseline must not lose its ``[B, 2] → [B, A] → [B]`` contract."""
    hints = get_type_hints(sample_masked_actions)

    assert hints["keys"] == BatchLegacyKey
    assert hints["action_mask"] == ActionMask
