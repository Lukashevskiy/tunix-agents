"""JAX-native contracts and adapters for CrafText training."""

from .contracts import RolloutBatch, Transition
from .rollout import collect_rollout

__all__ = ["RolloutBatch", "Transition", "collect_rollout"]
