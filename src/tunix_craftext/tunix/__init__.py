"""Tunix integration layer for topology, preflight and RLCluster workloads.

The package owns the production-shaped bridge to Tunix: role-to-mesh topology,
static preflight checks, workload configs, asset loading boundaries and cluster
construction. Root-level modules remain as compatibility shims while callers
migrate to this semantic package.
"""

from .preflight import (
    QwenTensorShape,
    pinned_qwen_tensor_shape,
    validate_agentic_grpo_preflight,
)
from .rlcluster_workload import (
    AgenticGrpoModelAssets,
    AgenticGrpoWorkloadSpec,
    PpoModelAssets,
    RLClusterWorkloadError,
    RLClusterWorkloadSpec,
    build_agentic_grpo_cluster,
    build_agentic_grpo_cluster_config,
    build_ppo_cluster,
    build_rlcluster_config,
    create_value_critic_from_actor,
    load_agentic_grpo_qwen_assets,
    load_ppo_gemma_assets,
    load_ppo_qwen_assets,
)
from .topology import (
    TopologyConfigError,
    TunixTopology,
    load_tunix_topology,
    role_to_meshes,
    tunix_role_to_meshes,
)

__all__ = [
    "AgenticGrpoModelAssets",
    "AgenticGrpoWorkloadSpec",
    "PpoModelAssets",
    "QwenTensorShape",
    "RLClusterWorkloadError",
    "RLClusterWorkloadSpec",
    "TopologyConfigError",
    "TunixTopology",
    "build_agentic_grpo_cluster",
    "build_agentic_grpo_cluster_config",
    "build_ppo_cluster",
    "build_rlcluster_config",
    "create_value_critic_from_actor",
    "load_agentic_grpo_qwen_assets",
    "load_ppo_gemma_assets",
    "load_ppo_qwen_assets",
    "load_tunix_topology",
    "pinned_qwen_tensor_shape",
    "role_to_meshes",
    "tunix_role_to_meshes",
    "validate_agentic_grpo_preflight",
]
