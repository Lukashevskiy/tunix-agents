"""Experience Builder contracts for Agentic PPO."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tunix_craftext.experience_builders import (
    PpoExperienceBuilder,
    UniversalMDPStep,
    broadcast_step_values_to_tokens,
    compute_mdp_gae,
)


def _universal_step(
    *,
    reward: tuple[float, float],
    value: tuple[float, float],
    step_mask: tuple[bool, bool] = (True, True),
    policy_token_mask: tuple[tuple[bool, bool], tuple[bool, bool]] | None = None,
) -> UniversalMDPStep:
    return UniversalMDPStep(
        prompt_tokens=jnp.asarray([[1, 2, 0], [3, 4, 5]], dtype=jnp.int32),
        prompt_mask=jnp.asarray([[True, True, False], [True, True, True]]),
        generation_tokens=jnp.asarray([[10, 11], [20, 0]], dtype=jnp.int32),
        generation_mask=jnp.asarray([[True, True], [True, False]]),
        actor_log_probs=jnp.asarray([[-0.2, -0.3], [-0.7, 0.0]], dtype=jnp.float32),
        step_mask=jnp.asarray(step_mask),
        reward=jnp.asarray(reward, dtype=jnp.float32),
        value=jnp.asarray(value, dtype=jnp.float32),
        policy_token_mask=(
            None if policy_token_mask is None else jnp.asarray(policy_token_mask)
        ),
        action_mask=jnp.asarray([[True, True], [True, False]]),
    )


def test_universal_mdp_step_validates_token_and_step_contract() -> None:
    step = _universal_step(reward=(1.0, 0.0), value=(0.5, 0.25))

    step.validate()

    assert step.actor_loss_token_mask.tolist() == [[True, True], [True, False]]


def test_universal_mdp_step_rejects_mismatched_actor_log_probs() -> None:
    step = _universal_step(reward=(1.0, 0.0), value=(0.5, 0.25))
    broken = UniversalMDPStep(
        prompt_tokens=step.prompt_tokens,
        generation_tokens=step.generation_tokens,
        generation_mask=step.generation_mask,
        actor_log_probs=jnp.zeros((2, 3), dtype=jnp.float32),
        step_mask=step.step_mask,
        reward=step.reward,
        value=step.value,
    )

    with pytest.raises(ValueError, match="actor_log_probs"):
        broken.validate()


def test_compute_mdp_gae_uses_time_axis_and_next_step_mask() -> None:
    rewards = jnp.asarray([[1.0], [2.0], [3.0]], dtype=jnp.float32)
    values = jnp.asarray([[0.5], [0.25], [0.0]], dtype=jnp.float32)
    step_masks = jnp.asarray([[True], [True], [False]])

    advantages, returns = compute_mdp_gae(
        rewards=rewards,
        values=values,
        step_masks=step_masks,
        gamma=1.0,
        gae_lambda=1.0,
    )

    # Last row is post-terminal padding.  Row 1 does not bootstrap from row 2
    # because step_masks[2] is false.
    assert advantages[:, 0].tolist() == pytest.approx([2.5, 1.75, 0.0])
    assert returns[:, 0].tolist() == pytest.approx([3.0, 2.0, 0.0])


def test_broadcast_step_values_to_tokens_masks_padding() -> None:
    token_values = broadcast_step_values_to_tokens(
        jnp.asarray([[1.5, -2.0]], dtype=jnp.float32),
        jnp.asarray([[[True, False], [True, True]]]),
    )

    np.testing.assert_allclose(
        np.asarray(token_values),
        np.asarray([[[1.5, 0.0], [-2.0, -2.0]]], dtype=np.float32),
    )


def test_ppo_experience_builder_broadcasts_mdp_advantages_to_valid_tokens() -> None:
    builder = PpoExperienceBuilder(gamma=1.0, gae_lambda=1.0)
    steps = (
        _universal_step(reward=(1.0, 0.0), value=(0.5, 0.0)),
        _universal_step(
            reward=(2.0, 1.0),
            value=(0.25, 0.5),
            step_mask=(True, False),
            policy_token_mask=((True, False), (True, True)),
        ),
    )

    experience = builder.build(steps)

    assert experience.prompt_tokens.shape == (4, 3)
    assert experience.generation_tokens.shape == (4, 2)
    assert experience.completion_mask.tolist() == [
        [True, True],
        [True, False],
        [True, False],
        [False, False],
    ]
    assert experience.step_advantages.tolist() == pytest.approx([2.5, 0.0, 1.75, 0.0])
    np.testing.assert_allclose(
        np.asarray(experience.advantages),
        np.asarray(
            [
                [2.5, 2.5],
                [0.0, 0.0],
                [1.75, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    assert experience.step_returns.tolist() == pytest.approx([3.0, 0.0, 2.0, 0.5])
    np.testing.assert_allclose(
        np.asarray(experience.returns),
        np.asarray(
            [
                [3.0, 3.0],
                [0.0, 0.0],
                [2.0, 0.0],
                [0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )


def test_ppo_experience_builder_requires_values() -> None:
    step = _universal_step(reward=(1.0, 0.0), value=(0.5, 0.25))
    missing_value = UniversalMDPStep(
        prompt_tokens=step.prompt_tokens,
        prompt_mask=step.prompt_mask,
        generation_tokens=step.generation_tokens,
        generation_mask=step.generation_mask,
        actor_log_probs=step.actor_log_probs,
        step_mask=step.step_mask,
        reward=step.reward,
    )

    with pytest.raises(ValueError, match="critic value"):
        PpoExperienceBuilder().build([missing_value])
