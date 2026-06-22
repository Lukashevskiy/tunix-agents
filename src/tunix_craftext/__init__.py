"""JAX-native contracts and adapters for CrafText training."""

from .adapters import CagedCrafTextAdapter, CrafTextAdapter, EnvironmentReset, EnvironmentStep
from .checkpoints import CheckpointMetadata, restore_checkpoint, save_checkpoint
from .config import ConfigError, MvpRunConfig, load_mvp_config
from .contracts import RolloutBatch, Transition
from .episode import collect_text_episode
from .interop import LoraAdapter, ModelTemplate, TensorRule, convert_state_dict, merge_lora_adapters
from .llm import LlmBackend, LlmRequest, LlmResponse, ScriptedLlmBackend
from .prompts import ActionCatalog, MegaPromptRenderer, PromptContext, RenderedPrompt
from .random_policy import ActionSamplingError, sample_masked_actions, validate_action_mask
from .replay import ReplayArtifact, ReplayStep, save_replay
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

__all__ = [
    "LoraAdapter",
    "CagedCrafTextAdapter",
    "CrafTextAdapter",
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
    "DecodedAction",
    "DecodeMetrics",
    "TextPolicy",
    "act",
    "decode_action",
    "decode_action_outcome",
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
    "ReplayArtifact",
    "ReplayStep",
    "save_replay",
    "CheckpointMetadata",
    "restore_checkpoint",
    "save_checkpoint",
]
