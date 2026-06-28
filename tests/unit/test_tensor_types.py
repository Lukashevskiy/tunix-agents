"""Regression tests for project-level jaxtyping public contracts."""

from typing import get_type_hints

from tunix_craftext.artifacts.text_trajectory import TextTrajectoryBatch
from tunix_craftext.core.contracts import RolloutBatch, Transition
from tunix_craftext.core.tensor_types import (
    ActionLogits,
    ActionMask,
    BatchBool,
    BatchFeatureFloat,
    BatchFloat,
    BatchLegacyKey,
    CausalInputPositionBatchInt,
    CausalInputTokenBatchInt,
    CausalLmHidden,
    CausalLmLogits,
    JaxArrayLike,
    JaxKey,
    ScalarInt,
    TimeBatchBool,
    TimeBatchFloat,
    TokenBatchBool,
    TokenBatchFloat,
    TokenBatchInt,
    TokenBatchLogits,
    ValueHeadKernel,
)
from tunix_craftext.models.llm_actor import LlmActorScores
from tunix_craftext.models.tunix_actor import CausalLmScoringModel, LinearValueHead
from tunix_craftext.research.learner import (
    ActorCritic,
    PromptConditionedTokenActorCritic,
    create_state,
    ppo_update,
)
from tunix_craftext.rollouts.random_policy import sample_masked_actions
from tunix_craftext.rollouts.reference import IndexedStepFn, PolicyFn, RolloutRecordT, StepFn


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


def test_reference_rollout_protocols_expose_batch_tensor_contracts() -> None:
    """Reference collectors should not accept unshaped policy/step scores."""
    policy_hints = get_type_hints(PolicyFn.__call__)
    step_hints = get_type_hints(StepFn.__call__)
    indexed_step_hints = get_type_hints(IndexedStepFn.__call__)

    assert policy_hints["return"].__args__[1:] == (BatchFloat, BatchFloat)
    assert step_hints["return"].__args__[2:] == (BatchFloat, BatchBool, BatchBool)
    assert indexed_step_hints["step_index"] == ScalarInt
    assert indexed_step_hints["return"].__args__[2:] == (BatchFloat, BatchBool, BatchBool)
    assert RolloutRecordT.__args__[2:] == (
        JaxArrayLike,
        JaxArrayLike,
        JaxArrayLike,
        JaxArrayLike,
        JaxArrayLike,
    )


def test_llm_actor_and_critic_scores_expose_token_axis_contracts() -> None:
    """Actor/critic score carriers must preserve token-level `[B, L]` axes."""
    actor_hints = get_type_hints(LlmActorScores)
    value_hints = get_type_hints(LinearValueHead)
    value_call_hints = get_type_hints(LinearValueHead.__call__)
    model_hints = get_type_hints(CausalLmScoringModel.__call__)

    assert actor_hints["token_logprobs"] == TokenBatchFloat
    assert actor_hints["values"] == TokenBatchFloat
    assert actor_hints["entropy"] == TokenBatchFloat
    assert actor_hints["token_mask"] == TokenBatchBool
    assert value_hints["kernel"] == ValueHeadKernel
    assert value_call_hints["return"] == TokenBatchFloat
    assert model_hints["input_tokens"] == CausalInputTokenBatchInt
    assert model_hints["positions"] == CausalInputPositionBatchInt
    assert model_hints["return"].__args__[0] == CausalLmLogits | CausalLmHidden


def test_research_learner_keeps_smoke_network_axes_explicit() -> None:
    """The research learner stays typed, even though it is not the production trainer."""
    actor_hints = get_type_hints(ActorCritic.__call__)
    token_actor_hints = get_type_hints(PromptConditionedTokenActorCritic.__call__)
    create_hints = get_type_hints(create_state)
    update_hints = get_type_hints(ppo_update)

    assert actor_hints["observation"] == BatchFeatureFloat
    assert actor_hints["return"].__args__ == (ActionLogits, BatchFloat)
    assert token_actor_hints["token_ids"] == TokenBatchInt
    assert token_actor_hints["return"].__args__ == (TokenBatchLogits, TokenBatchFloat)
    assert create_hints["key"] == JaxKey
    assert update_hints["observations"] == BatchFeatureFloat
