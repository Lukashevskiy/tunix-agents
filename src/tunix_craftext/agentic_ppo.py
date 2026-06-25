"""Agentic PPO learner bridge built on Tunix ``AgenticRLLearner``.

Tunix ships the async agentic rollout loop and critic-aware update scheduler in
``AgenticRLLearner``.  This module adds the missing PPO-shaped subclass for the
CrafText agentic path: trajectories are converted to PPO ``TrainExample``
objects with old policy log-probs, critic values, GAE advantages and returns.
The base learner then updates both actor and critic through public
``RLCluster.update_actor`` / ``RLCluster.update_critic``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import flax
import jax.numpy as jnp
import numpy as np

try:  # pragma: no cover - import availability is covered by compatibility tests.
    from tunix.rl import common, function_registry  # type: ignore[import-untyped]
    from tunix.rl import rl_cluster as rl_cluster_lib  # type: ignore[import-untyped]
    from tunix.rl import utils as rl_utils  # type: ignore[import-untyped]
    from tunix.rl.agentic import agentic_rl_learner  # type: ignore[import-untyped]
    from tunix.rl.agentic import utils as agentic_utils  # type: ignore[import-untyped]
except ImportError as error:  # pragma: no cover
    raise RuntimeError("install tunix-craftext[tunix] to use Agentic PPO") from error


TrainingInputT = agentic_rl_learner.TrainingInputT
RewardFn = agentic_rl_learner.RewardFn
MetricFn = agentic_rl_learner.MetricFn


@flax.struct.dataclass(frozen=True)
class AgenticPPOTrainExample(common.TrainExample):
    """PPO-shaped agentic train example consumed by actor and critic trainers."""

    returns: jnp.ndarray | None = None
    old_values: jnp.ndarray | None = None
    policy_version: np.ndarray | None = None


TrainExample = AgenticPPOTrainExample


@dataclasses.dataclass(kw_only=True)
class AgenticPPOConfig(agentic_rl_learner.AgenticRLConfig):
    """Configuration for critic-backed Agentic PPO.

    Unlike Agentic GRPO, this path does not need multiple generations per
    prompt. It uses the critic values returned by ``RLCluster.get_values`` and
    Tunix's registered ``gae`` advantage estimator / ``ppo`` policy and value
    losses.
    """

    algo_variant: str = "agentic_ppo"
    advantage_estimator: str = "gae"
    policy_loss_fn: str = "ppo"
    value_loss_fn: str = "ppo"
    num_generations: int = 1
    num_iterations: int = 1
    gamma: float = 1.0
    gae_lambda: float = 0.95
    beta: float = 0.04
    epsilon: float = 0.2
    epsilon_low: float | None = None
    epsilon_high: float | None = None
    epsilon_c: float | None = None
    entropy_coef: float | None = None
    clip_range_value: float = 0.2
    kl_method: str = "low_var_kl"
    kl_clamp_value: float | None = None

    def __post_init__(self) -> None:
        """Normalize PPO clipping bounds and reject unsupported contracts."""
        if self.num_generations != 1:
            raise ValueError("Agentic PPO expects exactly one generation per prompt")
        if self.num_iterations <= 0:
            raise ValueError("num_iterations must be positive")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if self.clip_range_value <= 0:
            raise ValueError("clip_range_value must be positive")
        if self.epsilon_c is not None and self.epsilon_c <= 1.0:
            raise ValueError("epsilon_c must be greater than 1 when enabled")
        if self.kl_method not in {"kl", "mse_kl", "low_var_kl"}:
            raise ValueError("kl_method must be one of kl, mse_kl or low_var_kl")
        self.epsilon_low = self.epsilon if self.epsilon_low is None else self.epsilon_low
        self.epsilon_high = self.epsilon if self.epsilon_high is None else self.epsilon_high


class AgenticPPOLearner(agentic_rl_learner.AgenticRLLearner[AgenticPPOConfig]):
    """Critic-backed PPO learner for Tunix agentic rollouts."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the base agentic loop and wire PPO actor/critic losses."""
        super().__init__(*args, **kwargs)
        configure_agentic_ppo_trainers(self.rl_cluster, self.algo_config)

    def _process_results(
        self,
        trajectories: list[Any],
        mode: rl_cluster_lib.Mode = rl_cluster_lib.Mode.TRAIN,
        expected_step: int | None = None,
    ) -> list[TrainExample]:
        """Convert one agentic trajectory group into a PPO train example."""
        if len(trajectories) != 1:
            raise ValueError("Agentic PPO expects one trajectory per prompt group")

        item = trajectories[0]
        traj = item.traj
        pad_id = self.rl_cluster.rollout.pad_id()
        eos_id = self.rl_cluster.rollout.eos_id()
        rollout_config = self.rl_cluster.cluster_config.rollout_config
        if isinstance(rollout_config, dict):
            rollout_config = rollout_config[mode]

        prompt_tokens = np.asarray(traj.get("prompt_tokens"), dtype=np.int32)
        completion_tokens = np.asarray(traj.get("conversation_tokens"), dtype=np.int32)
        completion_mask_raw = np.asarray(traj.get("conversation_masks"), dtype=np.int32)
        old_logprobs = traj.get("old_logprobs")
        policy_version = traj.get("policy_version")
        if policy_version is None:
            raise ValueError("policy_version is missing from trajectory task")

        padded_prompt, padded_completion, _ = agentic_utils.pad_prompt_and_completion(
            prompt_tokens,
            completion_tokens,
            rollout_config.max_prompt_length,
            self.algo_config.max_response_length,
            pad_id,
        )
        completion_ids = jnp.asarray([padded_completion[: self.algo_config.max_response_length]])
        completion_mask = jnp.asarray(
            [
                agentic_utils.right_pad(
                    completion_mask_raw,
                    self.algo_config.max_response_length,
                    0,
                )[: self.algo_config.max_response_length]
            ],
            dtype=jnp.bool_,
        )
        prompt_ids = jnp.asarray([padded_prompt])
        prompt_mask = prompt_ids != pad_id

        if self.algo_config.use_rollout_logps and old_logprobs is not None:
            old_per_token_logps = jnp.asarray(
                [
                    agentic_utils.right_pad(
                        np.asarray(old_logprobs, dtype=np.float32),
                        self.algo_config.max_response_length,
                        0.0,
                        dtype=np.float32,
                    )[: self.algo_config.max_response_length]
                ]
            )
        else:
            old_per_token_logps = self.rl_cluster.get_actor_per_token_logps(
                prompt_tokens=prompt_ids,
                completion_tokens=completion_ids,
                pad_id=pad_id,
                eos_id=eos_id,
                micro_batch_size=self.rl_cluster.cluster_config.training_config.compute_logps_micro_batch_size,
            )

        ref_per_token_logps = None
        if self.algo_config.beta != 0.0:
            ref_per_token_logps = self.rl_cluster.get_ref_per_token_logps(
                prompt_tokens=prompt_ids,
                completion_tokens=completion_ids,
                pad_id=pad_id,
                eos_id=eos_id,
                micro_batch_size=self.rl_cluster.cluster_config.training_config.compute_logps_micro_batch_size,
            )

        values = self.rl_cluster.get_values(
            prompt_tokens=prompt_ids,
            completion_tokens=completion_ids,
            pad_id=pad_id,
            eos_id=eos_id,
        )
        logits_to_keep = completion_ids.shape[1]
        values = values[:, -logits_to_keep - 1 : -1] * completion_mask

        original_inputs = rl_utils.merge_micro_batches([traj["original_input"]])
        reward_kwargs = {key: value for key, value in original_inputs.items() if key != "prompts"}
        reward_kwargs["trajectory_rewards"] = [traj.get("trajectory_reward", 0.0)]
        reward_scores = self._compute_rewards(
            prompts=original_inputs["prompts"],
            completions=[_assistant_completion_text(traj)],
            mode=mode,
            expected_step=expected_step,
            **reward_kwargs,
        )
        rewards = jnp.zeros_like(completion_ids, dtype=jnp.float32)
        eos_idx = jnp.maximum(completion_mask.astype(jnp.int32).sum(axis=-1) - 1, 0)
        rewards = rewards.at[jnp.arange(rewards.shape[0]), eos_idx].add(
            jnp.asarray(reward_scores, dtype=jnp.float32)
        )

        if ref_per_token_logps is not None:
            kl = common.compute_kl_divergence(
                old_per_token_logps,
                ref_per_token_logps,
                self.algo_config.kl_method,
                clamp_value=self.algo_config.kl_clamp_value,
            )
            rewards = rewards - self.algo_config.beta * kl * completion_mask

        advantage_estimator = function_registry.get_advantage_estimator(
            self.algo_config.advantage_estimator
        )
        advantages, returns = advantage_estimator(
            rewards=rewards,
            values=values,
            completion_mask=completion_mask,
            gamma=self.algo_config.gamma,
            gae_lambda=self.algo_config.gae_lambda,
        )

        self.rl_cluster.buffer_metrics_async(
            {
                "agentic_ppo/reward_mean": (float(np.mean(reward_scores)), np.mean),
                "agentic_ppo/value_mean": (float(jnp.mean(values)), np.mean),
                "agentic_ppo/return_mean": (float(jnp.mean(returns)), np.mean),
            },
            mode=mode,
            step=expected_step,
        )

        return [
            TrainExample(
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                completion_ids=completion_ids,
                completion_mask=completion_mask,
                ref_per_token_logps=ref_per_token_logps,
                advantages=advantages,
                old_per_token_logps=old_per_token_logps,
                returns=returns,
                old_values=values,
                policy_version=np.asarray([policy_version], dtype=np.int32),
            )
        ]


def configure_agentic_ppo_trainers(rl_cluster: Any, config: AgenticPPOConfig) -> None:
    """Attach Tunix PPO actor and value losses to an agentic ``RLCluster``."""
    if not getattr(getattr(rl_cluster, "inference_worker", None), "_models", {}).get("critic"):
        raise ValueError("Agentic PPO requires a critic model in RLCluster")

    policy_loss_fn = function_registry.get_policy_loss_fn(config.policy_loss_fn)
    value_loss_fn = function_registry.get_value_loss_fn(config.value_loss_fn)

    def actor_loss(model: object, train_example: object, algo_config: object) -> object:
        return policy_loss_fn(
            model,
            train_example,
            algo_config=algo_config,
            pad_id=rl_cluster.rollout.pad_id(),
            eos_id=rl_cluster.rollout.eos_id(),
            compute_logps_chunk_size=rl_cluster.cluster_config.training_config.compute_logps_chunk_size,
        )

    rl_cluster.actor_trainer.with_loss_fn(actor_loss, has_aux=True)
    rl_cluster.actor_trainer.with_gen_model_input_fn(
        lambda x: {"train_example": x, "algo_config": config}
    )
    rl_cluster.actor_trainer.with_rl_metrics_to_log(
        {"pg_clipfrac": np.mean, "kl": np.mean, "entropy": np.mean}
    )

    rl_cluster.critic_trainer.with_loss_fn(value_loss_fn, has_aux=True)
    rl_cluster.critic_trainer.with_gen_model_input_fn(
        lambda x: {
            "train_example": x,
            "clip_range_value": config.clip_range_value,
            "pad_id": rl_cluster.rollout.pad_id(),
            "eos_id": rl_cluster.rollout.eos_id(),
        }
    )
    rl_cluster.critic_trainer.with_rl_metrics_to_log(
        {"vpred_mean": np.mean, "vf_clipfrac": np.mean, "return_mean": np.mean}
    )


def _assistant_completion_text(traj: dict[str, Any]) -> str:
    conversation = traj.get("conversation_text") or []
    for message in conversation:
        if message.get("role") == "assistant":
            return str(message.get("content", ""))
    return ""


AgenticPpoConfig = AgenticPPOConfig
AgenticPpoLearner = AgenticPPOLearner
