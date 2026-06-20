"""JAX-native contracts and adapters for CrafText training."""

from .contracts import RolloutBatch, Transition
from .adapters import CagedCrafTextAdapter, CrafTextAdapter, EnvironmentReset, EnvironmentStep
from .interop import LoraAdapter, ModelTemplate, TensorRule, convert_state_dict, merge_lora_adapters
from .prompts import ActionCatalog, MegaPromptRenderer, PromptContext, RenderedPrompt
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
]
