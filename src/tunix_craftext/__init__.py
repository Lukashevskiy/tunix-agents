"""JAX-native contracts and adapters for CrafText training."""

from .contracts import RolloutBatch, Transition
from .adapters import CagedCrafTextAdapter, CrafTextAdapter, EnvironmentReset, EnvironmentStep
from .interop import LoraAdapter, ModelTemplate, TensorRule, convert_state_dict, merge_lora_adapters
from .prompts import ActionCatalog, MegaPromptRenderer, PromptContext, RenderedPrompt
from .text_policy import DecodedAction, DecodeMetrics, TextPolicy, act, decode_action
from .config import ConfigError, MvpRunConfig, load_mvp_config
from .runtime import CrafTextRuntime, build_craftext_runtime
from .random_policy import ActionSamplingError, sample_masked_actions, validate_action_mask
from .rollout import collect_rollout, collect_rollout_scan

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
    "ConfigError",
    "MvpRunConfig",
    "load_mvp_config",
    "CrafTextRuntime",
    "build_craftext_runtime",
    "ActionSamplingError",
    "sample_masked_actions",
    "validate_action_mask",
]
