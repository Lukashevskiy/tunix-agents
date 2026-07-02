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
from .artifacts.checkpoints import CheckpointMetadata, restore_checkpoint, save_checkpoint
from .artifacts.metric_pipeline import (
    LiveMetricPipeline,
    MetricComputationContext,
    MetricLoggerFactory,
    MetricPipelineError,
    MetricSource,
)
from .artifacts.observability import (
    ArtifactSink,
    CompositeArtifactSink,
    JsonlRunLogger,
    LoggerMethodMapping,
    MappedLoggerSink,
    MetricRecord,
    MetricSnapshotRecord,
    RunArtifact,
    ValidationTrajectoryRecord,
    checkpoint_artifact,
    flatten_scalar_metrics,
    optimizer_state_artifact,
    training_trajectory_artifact,
    validation_trajectory_artifact,
    validation_visualization_artifact,
    weights_artifact,
)
from .artifacts.profiling import PhaseProfiler, ProfileEvent, block_until_ready, save_profile
from .artifacts.replay import ReplayArtifact, ReplayStep, save_replay
from .artifacts.text_trajectory import (
    TextTrajectoryBatch,
    TextTrajectoryError,
    text_trajectory_from_replay,
)
from .artifacts.trajectory_gif import (
    frames_from_replay_payload,
    load_replay_payload,
    normalize_observation_image,
    scale_frame,
    write_gif,
)
from .core.contracts import RolloutBatch, Transition
from .env.config import ConfigError, MvpRunConfig, load_mvp_config
from .env.prompts import ActionCatalog, MegaPromptRenderer, PromptContext, RenderedPrompt
from .env.runtime import CrafTextRuntime, build_craftext_runtime
from .env.tasks import CrafTextTaskSampler, task_batches_from_craftext
from .env.text_policy import (
    DecodedAction,
    DecodeMetrics,
    TextPolicy,
    act,
    decode_action,
    decode_action_outcome,
)
from .interop import LoraAdapter, ModelTemplate, TensorRule, convert_state_dict, merge_lora_adapters
from .models.llm import LlmBackend, LlmRequest, LlmResponse, ScriptedLlmBackend
from .research.llm_ppo import (
    LlmPpoEvaluation,
    evaluate_llm_actor_critic_ppo,
    evaluate_separate_llm_actor_critic_ppo,
)
from .rollouts.hybrid import (
    HybridPpoStep,
    HybridPpoTrajectory,
    compute_masked_step_token_ppo_loss,
    hybrid_step_from_text_trajectory,
    hybrid_trajectory_from_steps,
    last_valid_token_values,
    shaped_step_rewards_from_text_trajectory,
)
from .rollouts.random_policy import ActionSamplingError, sample_masked_actions, validate_action_mask
from .rollouts.reference import collect_rollout, collect_rollout_scan
from .rollouts.text_episode import collect_text_episode
from .training.experience_builders import (
    ExperienceBuilder,
    PpoExperienceBuilder,
    TokenPPOExperience,
    UniversalMDPStep,
    broadcast_step_values_to_tokens,
    compute_mdp_gae,
)
from .tunix import (
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
    "ExperienceBuilder",
    "HybridPpoStep",
    "HybridPpoTrajectory",
    "PpoExperienceBuilder",
    "TokenPPOExperience",
    "UniversalMDPStep",
    "broadcast_step_values_to_tokens",
    "compute_masked_step_token_ppo_loss",
    "compute_mdp_gae",
    "hybrid_step_from_text_trajectory",
    "hybrid_trajectory_from_steps",
    "last_valid_token_values",
    "shaped_step_rewards_from_text_trajectory",
    "collect_text_episode",
    "collect_rollout",
    "collect_rollout_scan",
    "convert_state_dict",
    "merge_lora_adapters",
    "ActionCatalog",
    "MegaPromptRenderer",
    "PromptContext",
    "RenderedPrompt",
    "CrafTextTaskSampler",
    "task_batches_from_craftext",
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
    "frames_from_replay_payload",
    "load_replay_payload",
    "normalize_observation_image",
    "scale_frame",
    "write_gif",
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
    "CompositeArtifactSink",
    "JsonlRunLogger",
    "LoggerMethodMapping",
    "MappedLoggerSink",
    "MetricRecord",
    "MetricSnapshotRecord",
    "LiveMetricPipeline",
    "MetricComputationContext",
    "MetricLoggerFactory",
    "MetricPipelineError",
    "MetricSource",
    "RunArtifact",
    "ValidationTrajectoryRecord",
    "checkpoint_artifact",
    "flatten_scalar_metrics",
    "optimizer_state_artifact",
    "training_trajectory_artifact",
    "validation_trajectory_artifact",
    "validation_visualization_artifact",
    "weights_artifact",
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
