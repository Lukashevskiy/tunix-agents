"""JAX-native contracts and adapters for CrafText training."""

from .contracts import RolloutBatch, Transition
from .adapters import CagedCrafTextAdapter, CrafTextAdapter, EnvironmentReset, EnvironmentStep
from .interop import LoraAdapter, ModelTemplate, TensorRule, convert_state_dict, merge_lora_adapters
from .rollout import collect_rollout

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
    "convert_state_dict",
    "merge_lora_adapters",
]
