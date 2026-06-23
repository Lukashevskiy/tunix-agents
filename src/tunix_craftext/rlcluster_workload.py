"""Typed public-Tunix configuration for the actor/reference/critic workload."""

from __future__ import annotations

from dataclasses import dataclass

import optax

from .tunix_topology import TunixTopology, tunix_role_to_meshes


class RLClusterWorkloadError(ValueError):
    """Raised when a static RLCluster workload cannot have valid batch semantics."""


@dataclass(frozen=True)
class RLClusterWorkloadSpec:
    """Static PPO workload knobs shared by Tunix actor, critic and rollout roles."""

    max_steps: int
    eval_every_n_steps: int
    mini_batch_size: int
    train_micro_batch_size: int
    rollout_micro_batch_size: int
    max_prompt_length: int
    max_new_tokens: int
    kv_cache_size: int
    learning_rate: float = 1e-5

    def __post_init__(self) -> None:
        """Reject impossible static batch/cache contracts before accelerator allocation."""
        values = (
            self.max_steps,
            self.eval_every_n_steps,
            self.mini_batch_size,
            self.train_micro_batch_size,
            self.rollout_micro_batch_size,
            self.max_prompt_length,
            self.max_new_tokens,
            self.kv_cache_size,
        )
        if any(value <= 0 for value in values) or self.learning_rate <= 0:
            raise RLClusterWorkloadError("workload sizes and learning_rate must be positive")
        if self.mini_batch_size % self.train_micro_batch_size:
            raise RLClusterWorkloadError("mini_batch_size must divide train_micro_batch_size")
        if self.max_prompt_length + self.max_new_tokens > self.kv_cache_size:
            raise RLClusterWorkloadError("kv_cache_size must fit prompt plus generated tokens")


def build_rlcluster_config(topology: TunixTopology, spec: RLClusterWorkloadSpec) -> object:
    """Build the pinned public Tunix ``ClusterConfig`` without loading any model.

    Loading actor/reference/critic is intentionally a separate, hardware-gated
    operation. This boundary proves role placement and static training/rollout
    settings before HBM is allocated.
    """
    try:
        from tunix.rl.rl_cluster import (  # type: ignore[import-untyped]
            ClusterConfig,
            RLTrainingConfig,
        )
        from tunix.rl.rollout.base_rollout import RolloutConfig  # type: ignore[import-untyped]
    except ImportError as error:
        raise RLClusterWorkloadError(
            "install tunix-craftext[tunix] for RLCluster workload"
        ) from error
    training = RLTrainingConfig(
        max_steps=spec.max_steps,
        eval_every_n_steps=spec.eval_every_n_steps,
        actor_optimizer=optax.adamw(spec.learning_rate),
        critic_optimizer=optax.adamw(spec.learning_rate),
        mini_batch_size=spec.mini_batch_size,
        train_micro_batch_size=spec.train_micro_batch_size,
        rollout_micro_batch_size=spec.rollout_micro_batch_size,
    )
    rollout = RolloutConfig(
        max_prompt_length=spec.max_prompt_length,
        max_tokens_to_generate=spec.max_new_tokens,
        kv_cache_size=spec.kv_cache_size,
        return_logprobs=True,
        temperature=0.0,
    )
    return ClusterConfig(
        role_to_mesh=tunix_role_to_meshes(topology),
        rollout_engine="vanilla",
        training_config=training,
        rollout_config=rollout,
    )
