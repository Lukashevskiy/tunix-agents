"""Contracts for the Tunix Agentic PPO bridge."""

from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from tunix.rl import rl_cluster as rl_cluster_lib  # type: ignore[import-untyped]

from tunix_craftext.agentic_ppo import (
    AgenticPPOConfig,
    AgenticPPOLearner,
    configure_agentic_ppo_trainers,
    universal_mdp_steps_from_trajectory,
)
from tunix_craftext.experience_builders import PpoExperienceBuilder


class _Trainer:
    def __init__(self) -> None:
        self.loss_fn = None
        self.gen_model_input_fn = None
        self.metrics = None

    def with_loss_fn(self, fn, *, has_aux: bool) -> None:
        self.loss_fn = (fn, has_aux)

    def with_gen_model_input_fn(self, fn) -> None:
        self.gen_model_input_fn = fn

    def with_rl_metrics_to_log(self, metrics) -> None:
        self.metrics = metrics


class _Rollout:
    def pad_id(self) -> int:
        return 0

    def eos_id(self) -> int:
        return 2


class _FakeCluster:
    def __init__(self, *, has_critic: bool = True) -> None:
        self.actor_trainer = _Trainer()
        self.critic_trainer = _Trainer()
        self.rollout = _Rollout()
        self.inference_worker = SimpleNamespace(
            _models={"critic": object() if has_critic else None}
        )
        self.cluster_config = SimpleNamespace(
            training_config=SimpleNamespace(
                compute_logps_chunk_size=0,
                compute_logps_micro_batch_size=1,
            ),
            rollout_config=SimpleNamespace(max_prompt_length=4, max_tokens_to_generate=3),
        )
        self.metrics: list[tuple[dict, object, object]] = []
        self.actor_logprob_calls = 0
        self.ref_logprob_calls = 0

    def get_ref_per_token_logps(self, **kwargs):
        self.ref_logprob_calls += 1
        return jnp.zeros_like(kwargs["completion_tokens"], dtype=jnp.float32)

    def get_actor_per_token_logps(self, **kwargs):
        del kwargs
        self.actor_logprob_calls += 1
        return jnp.full((1, 3), -0.5, dtype=jnp.float32)

    def get_values(self, **kwargs):
        del kwargs
        return jnp.asarray([[0.0, 0.1, 0.2, 0.3]], dtype=jnp.float32)

    def buffer_metrics_async(self, metrics, *, mode, step) -> None:
        self.metrics.append((metrics, mode, step))


def test_agentic_ppo_config_defaults_to_single_generation_critic_path() -> None:
    config = AgenticPPOConfig(max_response_length=3)

    assert config.algo_variant == "agentic_ppo"
    assert config.num_generations == 1
    assert config.advantage_estimator == "gae"
    assert config.policy_loss_fn == "ppo"
    assert config.value_loss_fn == "ppo"
    assert config.epsilon_low == config.epsilon
    assert config.epsilon_high == config.epsilon


def test_agentic_ppo_config_rejects_grpo_style_groups() -> None:
    with pytest.raises(ValueError, match="exactly one generation"):
        AgenticPPOConfig(max_response_length=3, num_generations=2)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"num_iterations": 0}, "num_iterations"),
        ({"epsilon": 0.0}, "epsilon"),
        ({"clip_range_value": 0.0}, "clip_range_value"),
        ({"epsilon_c": 1.0}, "epsilon_c"),
        ({"kl_method": "wat"}, "kl_method"),
    ],
)
def test_agentic_ppo_config_rejects_invalid_optimizer_contracts(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        AgenticPPOConfig(max_response_length=3, **kwargs)


def test_configure_agentic_ppo_trainers_requires_critic_and_wires_losses() -> None:
    config = AgenticPPOConfig(max_response_length=3)
    cluster = _FakeCluster()

    configure_agentic_ppo_trainers(cluster, config)

    assert cluster.actor_trainer.loss_fn is not None
    assert cluster.actor_trainer.loss_fn[1] is True
    assert cluster.actor_trainer.gen_model_input_fn("batch") == {
        "train_example": "batch",
        "algo_config": config,
    }
    assert "pg_clipfrac" in cluster.actor_trainer.metrics
    assert cluster.critic_trainer.loss_fn is not None
    assert cluster.critic_trainer.gen_model_input_fn("batch")["clip_range_value"] == 0.2
    assert "return_mean" in cluster.critic_trainer.metrics


def test_configure_agentic_ppo_trainers_rejects_missing_critic() -> None:
    with pytest.raises(ValueError, match="requires a critic"):
        configure_agentic_ppo_trainers(_FakeCluster(has_critic=False), AgenticPPOConfig())


def test_agentic_ppo_process_results_adds_values_and_returns(monkeypatch) -> None:
    learner = object.__new__(AgenticPPOLearner)
    learner.rl_cluster = _FakeCluster()
    learner.algo_config = AgenticPPOConfig(max_response_length=3, beta=0.0)
    monkeypatch.setattr(
        learner,
        "_compute_rewards",
        lambda **kwargs: np.asarray([kwargs["trajectory_rewards"][0]], dtype=np.float32),
    )

    trajectory = SimpleNamespace(
        traj={
            "conversation_text": [{"role": "assistant", "content": "move"}],
            "prompt_tokens": np.asarray([9, 8], dtype=np.int32),
            "conversation_tokens": np.asarray([5, 6, 2], dtype=np.int32),
            "conversation_masks": np.asarray([1, 1, 1], dtype=np.int32),
            "old_logprobs": np.asarray([-0.2, -0.3, -0.4], dtype=np.float32),
            "policy_version": 4,
            "trajectory_reward": 1.25,
            "original_input": {"prompts": ["task"], "seed": np.asarray([7], dtype=np.int32)},
        }
    )

    [example] = learner._process_results([trajectory])

    assert example.prompt_ids.shape == (1, 4)
    assert example.completion_ids.shape == (1, 3)
    assert example.old_values.shape == (1, 3)
    assert example.returns.shape == (1, 3)
    assert example.advantages.shape == (1, 3)
    assert example.policy_version.tolist() == [4]
    assert bool(jnp.all(jnp.isfinite(example.returns)))
    assert learner.rl_cluster.metrics


def test_agentic_ppo_process_results_supports_rollout_config_by_mode(monkeypatch) -> None:
    learner = object.__new__(AgenticPPOLearner)
    learner.rl_cluster = _FakeCluster()
    learner.rl_cluster.cluster_config.rollout_config = {
        rl_cluster_lib.Mode.TRAIN: SimpleNamespace(
            max_prompt_length=5,
            max_tokens_to_generate=3,
        )
    }
    learner.algo_config = AgenticPPOConfig(max_response_length=3, beta=0.0)
    monkeypatch.setattr(learner, "_compute_rewards", lambda **kwargs: np.asarray([0.0]))

    trajectory = SimpleNamespace(
        traj={
            "conversation_text": [{"role": "assistant", "content": "move"}],
            "prompt_tokens": np.asarray([9, 8, 7], dtype=np.int32),
            "conversation_tokens": np.asarray([5], dtype=np.int32),
            "conversation_masks": np.asarray([1], dtype=np.int32),
            "policy_version": 4,
            "trajectory_reward": 0.0,
            "original_input": {"prompts": ["task"]},
        }
    )

    [example] = learner._process_results([trajectory], mode=rl_cluster_lib.Mode.TRAIN)

    assert example.prompt_ids.shape == (1, 5)


def test_agentic_ppo_process_results_rejects_grouped_grpo_trajectories() -> None:
    learner = object.__new__(AgenticPPOLearner)
    learner.rl_cluster = _FakeCluster()
    learner.algo_config = AgenticPPOConfig(max_response_length=3, beta=0.0)

    with pytest.raises(ValueError, match="one trajectory per prompt group"):
        learner._process_results([object(), object()])


def test_agentic_ppo_process_results_requires_policy_version(monkeypatch) -> None:
    learner = object.__new__(AgenticPPOLearner)
    learner.rl_cluster = _FakeCluster()
    learner.algo_config = AgenticPPOConfig(max_response_length=3, beta=0.0)
    monkeypatch.setattr(learner, "_compute_rewards", lambda **kwargs: np.asarray([0.0]))

    trajectory = SimpleNamespace(
        traj={
            "conversation_text": [{"role": "assistant", "content": "move"}],
            "prompt_tokens": np.asarray([9, 8], dtype=np.int32),
            "conversation_tokens": np.asarray([5, 6, 2], dtype=np.int32),
            "conversation_masks": np.asarray([1, 1, 1], dtype=np.int32),
            "trajectory_reward": 0.0,
            "original_input": {"prompts": ["task"]},
        }
    )

    with pytest.raises(ValueError, match="policy_version"):
        learner._process_results([trajectory])


def test_agentic_ppo_can_recompute_old_logprobs_from_actor(monkeypatch) -> None:
    learner = object.__new__(AgenticPPOLearner)
    learner.rl_cluster = _FakeCluster()
    learner.algo_config = AgenticPPOConfig(
        max_response_length=3,
        beta=0.0,
        use_rollout_logps=False,
    )
    monkeypatch.setattr(learner, "_compute_rewards", lambda **kwargs: np.asarray([0.0]))

    trajectory = SimpleNamespace(
        traj={
            "conversation_text": [{"role": "assistant", "content": "move"}],
            "prompt_tokens": np.asarray([9, 8], dtype=np.int32),
            "conversation_tokens": np.asarray([5, 6, 2], dtype=np.int32),
            "conversation_masks": np.asarray([1, 1, 1], dtype=np.int32),
            "old_logprobs": np.asarray([-9.0, -9.0, -9.0], dtype=np.float32),
            "policy_version": 4,
            "trajectory_reward": 0.0,
            "original_input": {"prompts": ["task"]},
        }
    )

    [example] = learner._process_results([trajectory])

    assert learner.rl_cluster.actor_logprob_calls == 1
    assert example.old_per_token_logps.tolist() == [[-0.5, -0.5, -0.5]]


def test_agentic_ppo_process_results_applies_reference_kl_when_beta_enabled(
    monkeypatch,
) -> None:
    learner = object.__new__(AgenticPPOLearner)
    learner.rl_cluster = _FakeCluster()
    learner.algo_config = AgenticPPOConfig(max_response_length=3, beta=0.2, kl_method="kl")
    monkeypatch.setattr(learner, "_compute_rewards", lambda **kwargs: np.asarray([1.0]))

    trajectory = SimpleNamespace(
        traj={
            "conversation_text": [{"role": "assistant", "content": "move"}],
            "prompt_tokens": np.asarray([9, 8], dtype=np.int32),
            "conversation_tokens": np.asarray([5, 6, 2], dtype=np.int32),
            "conversation_masks": np.asarray([1, 1, 1], dtype=np.int32),
            "old_logprobs": np.asarray([-0.2, -0.3, -0.4], dtype=np.float32),
            "policy_version": 4,
            "trajectory_reward": 1.0,
            "original_input": {"prompts": ["task"]},
        }
    )

    [example] = learner._process_results([trajectory])

    assert learner.rl_cluster.ref_logprob_calls == 1
    assert example.ref_per_token_logps is not None
    assert example.ref_per_token_logps.shape == example.completion_ids.shape
    assert bool(jnp.all(jnp.isfinite(example.returns)))


def test_agentic_ppo_process_results_uses_mdp_steps_and_experience_builder(
    monkeypatch,
) -> None:
    learner = object.__new__(AgenticPPOLearner)
    learner.rl_cluster = _FakeCluster()
    learner.algo_config = AgenticPPOConfig(
        max_response_length=3,
        beta=0.0,
        gamma=1.0,
        gae_lambda=1.0,
    )
    monkeypatch.setattr(
        learner,
        "_compute_rewards",
        lambda **kwargs: pytest.fail("MDP step rewards must bypass EOS reward fallback"),
    )

    trajectory = SimpleNamespace(
        traj={
            "policy_version": 9,
            "original_input": {"prompts": ["task"]},
            "mdp_steps": [
                {
                    "prompt_tokens": np.asarray([9, 8], dtype=np.int32),
                    "generation_tokens": np.asarray([5, 6], dtype=np.int32),
                    "generation_mask": np.asarray([1, 1], dtype=np.int32),
                    "actor_log_probs": np.asarray([-0.2, -0.3], dtype=np.float32),
                    "reward": 1.0,
                    "value": 0.5,
                    "step_mask": True,
                },
                {
                    "prompt_tokens": np.asarray([9, 8, 7], dtype=np.int32),
                    "generation_tokens": np.asarray([2], dtype=np.int32),
                    "generation_mask": np.asarray([1], dtype=np.int32),
                    "actor_log_probs": np.asarray([-0.4], dtype=np.float32),
                    "reward": 2.0,
                    "value": 0.25,
                    "step_mask": True,
                },
            ],
        }
    )

    [example] = learner._process_results([trajectory])

    assert example.prompt_ids.shape == (2, 4)
    assert example.completion_ids.shape == (2, 3)
    assert example.completion_mask.tolist() == [[True, True, False], [True, False, False]]
    np.testing.assert_allclose(
        np.asarray(example.old_per_token_logps),
        np.asarray([[-0.2, -0.3, 0.0], [-0.4, 0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(example.advantages),
        np.asarray([[2.5, 2.5, 0.0], [1.75, 0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(example.returns),
        np.asarray([[3.0, 3.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        np.asarray(example.old_values),
        np.asarray([[0.5, 0.5, 0.0], [0.25, 0.0, 0.0]], dtype=np.float32),
    )
    assert example.policy_version.tolist() == [9, 9]
    assert learner.rl_cluster.metrics[-1][0]["agentic_ppo/mdp_return_mean"][0] == pytest.approx(
        2.5
    )


def test_agentic_ppo_process_results_mdp_steps_can_fetch_reference_logps(
    monkeypatch,
) -> None:
    learner = object.__new__(AgenticPPOLearner)
    learner.rl_cluster = _FakeCluster()
    learner.algo_config = AgenticPPOConfig(max_response_length=2, beta=0.1)
    monkeypatch.setattr(
        learner,
        "_compute_rewards",
        lambda **kwargs: pytest.fail("MDP step rewards must bypass EOS reward fallback"),
    )

    trajectory = SimpleNamespace(
        traj={
            "policy_version": 3,
            "mdp_steps": [
                {
                    "prompt_tokens": [9],
                    "generation_tokens": [5, 6],
                    "generation_mask": [1, 1],
                    "actor_log_probs": [-0.2, -0.3],
                    "reward": 1.0,
                    "value": 0.0,
                    "step_mask": True,
                }
            ],
        }
    )

    [example] = learner._process_results([trajectory])

    assert learner.rl_cluster.ref_logprob_calls == 1
    assert example.ref_per_token_logps is not None
    assert example.ref_per_token_logps.shape == example.completion_ids.shape


def test_universal_mdp_steps_from_trajectory_pads_variable_lengths() -> None:
    steps = universal_mdp_steps_from_trajectory(
        {
            "mdp_steps": [
                {
                    "prompt_tokens": [1, 2, 3],
                    "assistant_tokens": [4],
                    "assistant_masks": [1],
                    "logprobs": [-0.1],
                    "reward": 1.0,
                    "value": 0.5,
                }
            ]
        },
        max_prompt_length=5,
        max_response_length=3,
        pad_id=0,
    )

    [step] = steps
    assert step.prompt_tokens.tolist() == [[1, 2, 3, 0, 0]]
    assert step.prompt_mask.tolist() == [[True, True, True, False, False]]
    assert step.generation_tokens.tolist() == [[4, 0, 0]]
    assert step.generation_mask.tolist() == [[True, False, False]]
    np.testing.assert_allclose(
        np.asarray(step.actor_log_probs),
        np.asarray([[-0.1, 0.0, 0.0]], dtype=np.float32),
    )
    assert step.reward.tolist() == [1.0]
    assert step.value.tolist() == [0.5]


def test_universal_mdp_steps_from_trajectory_supports_batched_rows_and_action_mask() -> None:
    steps = universal_mdp_steps_from_trajectory(
        {
            "mdp_steps": [
                {
                    "prompt_token_ids": [[1, 2], [3, 0]],
                    "completion_tokens": [[4, 5], [6, 0]],
                    "completion_mask": [[1, 1], [1, 0]],
                    "old_logprobs": [[-0.1, -0.2], [-0.3, 0.0]],
                    "policy_token_mask": [[1, 0], [1, 0]],
                    "action_mask": [[1, 0, 1], [0, 1, 1]],
                    "reward": [1.0, -1.0],
                    "value": [0.5, -0.25],
                    "step_mask": [True, False],
                }
            ]
        },
        max_prompt_length=3,
        max_response_length=2,
        pad_id=0,
    )

    [step] = steps
    assert step.prompt_tokens.tolist() == [[1, 2, 0], [3, 0, 0]]
    assert step.generation_mask.tolist() == [[True, True], [True, False]]
    assert step.policy_token_mask is not None
    assert step.actor_loss_token_mask.tolist() == [[True, False], [True, False]]
    built = PpoExperienceBuilder().build(steps)
    assert built.completion_mask.tolist() == [[True, False], [False, False]]
    assert step.action_mask is not None
    assert step.action_mask.tolist() == [[True, False, True], [False, True, True]]
    assert step.reward.tolist() == pytest.approx([1.0, -1.0])
    assert step.value.tolist() == pytest.approx([0.5, -0.25])
    assert step.step_mask.tolist() == [True, False]


@pytest.mark.parametrize(
    ("raw_step", "message"),
    [
        ({}, "missing required field"),
        (
            {
                "prompt_tokens": [[[1]]],
                "generation_tokens": [2],
                "reward": 1.0,
                "value": 0.0,
            },
            "token arrays",
        ),
        (
            {
                "prompt_tokens": [[1], [2]],
                "generation_tokens": [[3]],
                "reward": 1.0,
                "value": 0.0,
            },
            "batch size",
        ),
        (
            {
                "prompt_tokens": [1],
                "generation_tokens": [2],
                "actor_log_probs": [[-0.1], [-0.2]],
                "reward": 1.0,
                "value": 0.0,
            },
            "logprob batch size",
        ),
        (
            {
                "prompt_tokens": [1],
                "generation_tokens": [2],
                "reward": [1.0, 2.0],
                "value": 0.0,
            },
            "scalar or shape",
        ),
    ],
)
def test_universal_mdp_steps_from_trajectory_rejects_invalid_evidence(
    raw_step: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        universal_mdp_steps_from_trajectory(
            {"mdp_steps": [raw_step]},
            max_prompt_length=2,
            max_response_length=2,
            pad_id=0,
        )


def test_universal_mdp_steps_from_trajectory_rejects_empty_steps() -> None:
    with pytest.raises(ValueError, match="at least one step"):
        universal_mdp_steps_from_trajectory(
            {"mdp_steps": []},
            max_prompt_length=2,
            max_response_length=2,
            pad_id=0,
        )
