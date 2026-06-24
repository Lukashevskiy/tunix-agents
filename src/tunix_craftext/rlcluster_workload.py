"""Typed public-Tunix workload configurations without project-local scheduling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import optax  # type: ignore[import-untyped]

from .tunix_adapter import load_qwen_model_on_mesh, load_qwen_tokenizer
from .tunix_topology import TunixTopology, role_to_meshes, tunix_role_to_meshes


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


@dataclass(frozen=True)
class AgenticGrpoWorkloadSpec(RLClusterWorkloadSpec):
    """Static workload knobs for Tunix Agentic GRPO actor/rollout/reference roles."""

    num_generations: int = 2
    max_concurrency: int = 1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.num_generations < 2:
            raise RLClusterWorkloadError("Agentic GRPO requires at least two generations")
        if self.max_concurrency <= 0:
            raise RLClusterWorkloadError("max_concurrency must be positive")


@dataclass(frozen=True)
class AgenticGrpoModelAssets:
    """Trainable actor, frozen reference and tokenizer loaded for one GRPO cluster."""

    actor: object
    reference: object
    tokenizer: object


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


def build_agentic_grpo_cluster_config(
    topology: TunixTopology, spec: AgenticGrpoWorkloadSpec
) -> object:
    """Build the public Tunix config for the actor/rollout/reference Agentic GRPO path."""
    if "critic" in topology.role_to_device_indices:
        raise RLClusterWorkloadError("Agentic GRPO topology must not declare a critic role")
    try:
        from tunix.rl.rl_cluster import (  # type: ignore[import-untyped]
            ClusterConfig,
            RLTrainingConfig,
        )
        from tunix.rl.rollout.base_rollout import RolloutConfig  # type: ignore[import-untyped]
    except ImportError as error:
        raise RLClusterWorkloadError(
            "install tunix-craftext[tunix] for Agentic GRPO workload"
        ) from error
    training = RLTrainingConfig(
        max_steps=spec.max_steps,
        eval_every_n_steps=spec.eval_every_n_steps,
        actor_optimizer=optax.adamw(spec.learning_rate),
        mini_batch_size=spec.mini_batch_size,
        train_micro_batch_size=spec.train_micro_batch_size,
        rollout_micro_batch_size=spec.rollout_micro_batch_size,
        compute_logps_micro_batch_size=spec.train_micro_batch_size,
    )
    rollout = RolloutConfig(
        max_prompt_length=spec.max_prompt_length,
        max_tokens_to_generate=spec.max_new_tokens,
        kv_cache_size=spec.kv_cache_size,
        return_logprobs=True,
        temperature=1.0,
    )
    return ClusterConfig(
        role_to_mesh=tunix_role_to_meshes(topology),
        rollout_engine="vanilla",
        training_config=training,
        rollout_config=rollout,
    )


def load_agentic_grpo_qwen_assets(
    snapshot: Path, topology: TunixTopology
) -> AgenticGrpoModelAssets:
    """Load independent Qwen actor/reference copies on their declared Tunix meshes."""
    if "critic" in topology.role_to_device_indices:
        raise RLClusterWorkloadError("Agentic GRPO topology must not declare a critic role")
    meshes = role_to_meshes(topology)
    return AgenticGrpoModelAssets(
        actor=load_qwen_model_on_mesh(snapshot, meshes["actor"], dtype=jnp.float32),
        reference=load_qwen_model_on_mesh(snapshot, meshes["reference"], dtype=jnp.bfloat16),
        tokenizer=load_qwen_tokenizer(snapshot),
    )


def build_agentic_grpo_cluster(
    topology: TunixTopology,
    spec: AgenticGrpoWorkloadSpec,
    assets: AgenticGrpoModelAssets,
) -> object:
    """Create the real public Tunix ``RLCluster`` from already loaded GRPO assets."""
    try:
        from tunix.rl.rl_cluster import RLCluster  # type: ignore[import-untyped]
    except ImportError as error:
        raise RLClusterWorkloadError(
            "install tunix-craftext[tunix] for Agentic GRPO workload"
        ) from error
    return RLCluster(
        actor=assets.actor,
        reference=assets.reference,
        tokenizer=assets.tokenizer,
        cluster_config=build_agentic_grpo_cluster_config(topology, spec),
    )
