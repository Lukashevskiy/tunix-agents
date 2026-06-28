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

from .experience_builders import PpoExperienceBuilder, UniversalMDPStep

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

        policy_version = traj.get("policy_version")
        if policy_version is None:
            raise ValueError("policy_version is missing from trajectory task")
        if traj.get("mdp_steps"):
            return [
                self._process_mdp_steps(
                    traj,
                    rollout_config=rollout_config,
                    pad_id=pad_id,
                    eos_id=eos_id,
                    policy_version=policy_version,
                    mode=mode,
                    expected_step=expected_step,
                )
            ]

        prompt_tokens = np.asarray(traj.get("prompt_tokens"), dtype=np.int32)
        completion_tokens = np.asarray(traj.get("conversation_tokens"), dtype=np.int32)
        completion_mask_raw = np.asarray(traj.get("conversation_masks"), dtype=np.int32)
        old_logprobs = traj.get("old_logprobs")

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

    def _process_mdp_steps(
        self,
        traj: dict[str, Any],
        *,
        rollout_config: Any,
        pad_id: int,
        eos_id: int,
        policy_version: int,
        mode: rl_cluster_lib.Mode,
        expected_step: int | None,
    ) -> TrainExample:
        """Convert rich per-MDP-step evidence via ``PpoExperienceBuilder``."""
        steps = universal_mdp_steps_from_trajectory(
            traj,
            max_prompt_length=rollout_config.max_prompt_length,
            max_response_length=self.algo_config.max_response_length,
            pad_id=pad_id,
        )
        experience = PpoExperienceBuilder(
            gamma=self.algo_config.gamma,
            gae_lambda=self.algo_config.gae_lambda,
        ).build(steps)

        ref_per_token_logps = None
        if self.algo_config.beta != 0.0:
            ref_per_token_logps = self.rl_cluster.get_ref_per_token_logps(
                prompt_tokens=experience.prompt_tokens,
                completion_tokens=experience.generation_tokens,
                pad_id=pad_id,
                eos_id=eos_id,
                micro_batch_size=self.rl_cluster.cluster_config.training_config.compute_logps_micro_batch_size,
            )

        old_values = jnp.asarray(experience.step_values, dtype=jnp.float32)[:, None] * jnp.asarray(
            experience.completion_mask,
            dtype=jnp.float32,
        )
        self.rl_cluster.buffer_metrics_async(
            {
                "agentic_ppo/mdp_reward_mean": (
                    float(jnp.mean(experience.step_rewards)),
                    np.mean,
                ),
                "agentic_ppo/mdp_value_mean": (
                    float(jnp.mean(experience.step_values)),
                    np.mean,
                ),
                "agentic_ppo/mdp_return_mean": (
                    float(jnp.mean(experience.step_returns)),
                    np.mean,
                ),
            },
            mode=mode,
            step=expected_step,
        )

        return TrainExample(
            prompt_ids=experience.prompt_tokens,
            prompt_mask=experience.prompt_mask,
            completion_ids=experience.generation_tokens,
            completion_mask=experience.completion_mask,
            ref_per_token_logps=ref_per_token_logps,
            advantages=experience.advantages,
            old_per_token_logps=experience.old_per_token_logps,
            returns=experience.returns,
            old_values=old_values,
            policy_version=np.full(
                (experience.generation_tokens.shape[0],),
                policy_version,
                dtype=np.int32,
            ),
        )


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


def universal_mdp_steps_from_trajectory(
    traj: dict[str, Any],
    *,
    max_prompt_length: int,
    max_response_length: int,
    pad_id: int,
) -> tuple[UniversalMDPStep, ...]:
    """Build padded universal MDP steps from rich agentic trajectory evidence.

    Each item in ``traj["mdp_steps"]`` may provide 1D arrays for one environment
    or already batched 2D arrays.  Variable lengths are right-padded to the
    rollout profile limits before the Experience Builder stacks time.
    """
    raw_steps = traj.get("mdp_steps") or ()
    if not raw_steps:
        raise ValueError("mdp_steps must contain at least one step")
    return tuple(
        _universal_mdp_step_from_mapping(
            raw_step,
            max_prompt_length=max_prompt_length,
            max_response_length=max_response_length,
            pad_id=pad_id,
        )
        for raw_step in raw_steps
    )


def _universal_mdp_step_from_mapping(
    raw_step: Any,
    *,
    max_prompt_length: int,
    max_response_length: int,
    pad_id: int,
) -> UniversalMDPStep:
    step = dict(raw_step)
    prompt_tokens = _as_batched_int_array(_first_present(step, "prompt_tokens", "prompt_token_ids"))
    generation_tokens = _as_batched_int_array(
        _first_present(step, "generation_tokens", "completion_tokens", "assistant_tokens")
    )
    batch_size = prompt_tokens.shape[0]
    if generation_tokens.shape[0] != batch_size:
        raise ValueError("generation token batch size must match prompt batch size")

    prompt_tokens, prompt_mask = _right_pad_2d(
        prompt_tokens,
        length=max_prompt_length,
        pad_value=pad_id,
        dtype=np.int32,
    )
    generation_tokens, default_generation_mask = _right_pad_2d(
        generation_tokens,
        length=max_response_length,
        pad_value=pad_id,
        dtype=np.int32,
    )
    generation_mask = _optional_mask(
        step,
        ("generation_mask", "completion_mask", "assistant_masks"),
        default_generation_mask,
        length=max_response_length,
    )
    actor_log_probs = _optional_float_tokens(
        step,
        ("actor_log_probs", "old_logprobs", "logprobs"),
        batch_size=batch_size,
        length=max_response_length,
    )
    policy_token_mask = _optional_mask(
        step,
        ("policy_token_mask",),
        generation_mask,
        length=max_response_length,
    )
    return UniversalMDPStep(
        prompt_tokens=jnp.asarray(prompt_tokens),
        prompt_mask=jnp.asarray(prompt_mask, dtype=bool),
        generation_tokens=jnp.asarray(generation_tokens),
        generation_mask=jnp.asarray(generation_mask, dtype=bool),
        actor_log_probs=jnp.asarray(actor_log_probs, dtype=jnp.float32),
        step_mask=jnp.asarray(_batch_vector(step.get("step_mask", True), batch_size, bool)),
        reward=jnp.asarray(_batch_vector(_first_present(step, "reward"), batch_size, np.float32)),
        value=jnp.asarray(_batch_vector(_first_present(step, "value"), batch_size, np.float32)),
        policy_token_mask=jnp.asarray(policy_token_mask, dtype=bool),
        action_mask=(
            None
            if step.get("action_mask") is None
            else jnp.asarray(step["action_mask"], dtype=bool)
        ),
    )


def _first_present(mapping: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    raise ValueError(f"missing required field: {'/'.join(keys)}")


def _as_batched_int_array(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.int32)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2:
        raise ValueError("token arrays must have shape [L] or [B, L]")
    return array


def _right_pad_2d(
    value: np.ndarray,
    *,
    length: int,
    pad_value: int | float,
    dtype: Any,
) -> tuple[np.ndarray, np.ndarray]:
    if length <= 0:
        raise ValueError("pad length must be positive")
    array = value.astype(dtype)
    clipped = array[:, :length]
    mask = np.ones(clipped.shape, dtype=bool)
    if clipped.shape[1] < length:
        width = length - clipped.shape[1]
        clipped = np.pad(clipped, ((0, 0), (0, width)), constant_values=pad_value)
        mask = np.pad(mask, ((0, 0), (0, width)), constant_values=False)
    return clipped.astype(dtype), mask


def _optional_mask(
    mapping: dict[str, Any],
    keys: tuple[str, ...],
    default: np.ndarray,
    *,
    length: int,
) -> np.ndarray:
    value = next((mapping[key] for key in keys if mapping.get(key) is not None), None)
    if value is None:
        return default
    array = np.asarray(value, dtype=bool)
    if array.ndim == 1:
        array = array[None, :]
    padded, _ = _right_pad_2d(array.astype(np.int32), length=length, pad_value=0, dtype=np.int32)
    return padded.astype(bool)


def _optional_float_tokens(
    mapping: dict[str, Any],
    keys: tuple[str, ...],
    *,
    batch_size: int,
    length: int,
) -> np.ndarray:
    value = next((mapping[key] for key in keys if mapping.get(key) is not None), None)
    if value is None:
        return np.zeros((batch_size, length), dtype=np.float32)
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[None, :]
    if array.shape[0] != batch_size:
        raise ValueError("token logprob batch size must match prompt batch size")
    padded, _ = _right_pad_2d(array, length=length, pad_value=0.0, dtype=np.float32)
    return padded


def _batch_vector(value: Any, batch_size: int, dtype: Any) -> np.ndarray:
    array = np.asarray(value, dtype=dtype)
    if array.ndim == 0:
        array = np.full((batch_size,), array.item(), dtype=dtype)
    if array.shape != (batch_size,):
        raise ValueError(f"value must be scalar or shape ({batch_size},), got {array.shape}")
    return array


AgenticPpoConfig = AgenticPPOConfig
AgenticPpoLearner = AgenticPPOLearner
