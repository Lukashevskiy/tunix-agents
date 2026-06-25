"""JAX-native contracts and adapters for CrafText training.

This package exposes a compact, typed interface for transforming CrafText-family
vendor environments into a training-safe JAX workflow. Public exports include
protocols, runtime adapters, prompt rendering helpers, and replay/trajectory
contracts for reproducible experiments.
"""

from .adapters import (
    CagedCrafTextAdapter,
    CraftaxAdapter,
    CrafTextAdapter,
    CrafTextEpisodeContext,
    EnvironmentReset,
    EnvironmentStep,
)
from .checkpoints import CheckpointMetadata, restore_checkpoint, save_checkpoint
from .config import ConfigError, MvpRunConfig, load_mvp_config
from .contracts import RolloutBatch, Transition
from .episode import collect_text_episode
from .interop import LoraAdapter, ModelTemplate, TensorRule, convert_state_dict, merge_lora_adapters
from .llm import LlmBackend, LlmRequest, LlmResponse, ScriptedLlmBackend
from .observability import (
    ArtifactSink,
    JsonlRunLogger,
    MetricRecord,
    RunArtifact,
    ValidationTrajectoryRecord,
)
from .profiling import PhaseProfiler, ProfileEvent, block_until_ready, save_profile
from .prompts import ActionCatalog, MegaPromptRenderer, PromptContext, RenderedPrompt
from .random_policy import ActionSamplingError, sample_masked_actions, validate_action_mask
from .replay import ReplayArtifact, ReplayStep, save_replay
from .research.llm_ppo import (
    LlmPpoEvaluation,
    evaluate_llm_actor_critic_ppo,
    evaluate_separate_llm_actor_critic_ppo,
)
from .rollout import collect_rollout, collect_rollout_scan
from .runtime import CrafTextRuntime, build_craftext_runtime
from .text_policy import (
    DecodedAction,
    DecodeMetrics,
    TextPolicy,
    act,
    decode_action,
    decode_action_outcome,
)
from .text_trajectory import TextTrajectoryBatch, TextTrajectoryError, text_trajectory_from_replay
from .tunix_topology import (
    TopologyConfigError,
    TunixTopology,
    load_tunix_topology,
    role_to_meshes,
    tunix_role_to_meshes,
)

__all__ = [
    "LoraAdapter",
    "CagedCrafTextAdapter",
    "CraftaxAdapter",
    "CrafTextAdapter",
    "CrafTextEpisodeContext",
    "EnvironmentReset",
    "EnvironmentStep",
    "ModelTemplate",
    "RolloutBatch",
    "TensorRule",
    "Transition",
    "collect_text_episode",
    "collect_rollout",
    "collect_rollout_scan",
    "convert_state_dict",
    "merge_lora_adapters",
    "ActionCatalog",
    "MegaPromptRenderer",
    "PromptContext",
    "RenderedPrompt",
    "PhaseProfiler",
    "ProfileEvent",
    "block_until_ready",
    "save_profile",
    "DecodedAction",
    "DecodeMetrics",
    "TextPolicy",
    "act",
    "decode_action",
    "decode_action_outcome",
    "TopologyConfigError",
    "TunixTopology",
    "load_tunix_topology",
    "role_to_meshes",
    "tunix_role_to_meshes",
    "TextTrajectoryBatch",
    "TextTrajectoryError",
    "text_trajectory_from_replay",
    "ConfigError",
    "MvpRunConfig",
    "load_mvp_config",
    "CrafTextRuntime",
    "build_craftext_runtime",
    "ActionSamplingError",
    "sample_masked_actions",
    "validate_action_mask",
    "LlmBackend",
    "LlmRequest",
    "LlmResponse",
    "ScriptedLlmBackend",
    "ArtifactSink",
    "JsonlRunLogger",
    "MetricRecord",
    "RunArtifact",
    "ValidationTrajectoryRecord",
    "LlmPpoEvaluation",
    "evaluate_llm_actor_critic_ppo",
    "evaluate_separate_llm_actor_critic_ppo",
    "ReplayArtifact",
    "ReplayStep",
    "save_replay",
    "CheckpointMetadata",
    "restore_checkpoint",
    "save_checkpoint",
]
