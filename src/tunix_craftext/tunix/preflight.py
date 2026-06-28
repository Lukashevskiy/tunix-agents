"""Static mesh/model/batch checks performed before Qwen weights enter memory."""

from __future__ import annotations

from dataclasses import dataclass

from .rlcluster_workload import AgenticGrpoWorkloadSpec, RLClusterWorkloadError
from .topology import TunixTopology


@dataclass(frozen=True)
class QwenTensorShape:
    """Minimal tensor dimensions needed for mesh divisibility validation."""

    embed_dim: int
    num_heads: int
    vocab_size: int


def pinned_qwen_tensor_shape() -> QwenTensorShape:
    """Read the pinned Tunix Qwen configuration without loading model weights."""
    try:
        from tunix.models.automodel import call_model_config  # type: ignore[import-untyped]
    except ImportError as error:
        raise RLClusterWorkloadError("install tunix-craftext[tunix] for Qwen preflight") from error
    config = call_model_config("qwen2.5-0.5b")
    return QwenTensorShape(config.embed_dim, config.num_heads, config.vocab_size)


def validate_agentic_grpo_preflight(
    topology: TunixTopology,
    spec: AgenticGrpoWorkloadSpec,
    shape: QwenTensorShape,
) -> None:
    """Reject impossible role meshes and static workload dimensions before model load."""
    if shape.embed_dim <= 0 or shape.num_heads <= 0 or shape.vocab_size <= 0:
        raise RLClusterWorkloadError("Qwen tensor dimensions must be positive")
    for role in ("actor", "rollout", "reference"):
        degree = len(topology.role_to_device_indices[role])
        for name, value in (
            ("num_heads", shape.num_heads),
            ("embed_dim", shape.embed_dim),
            ("vocab_size", shape.vocab_size),
        ):
            if value % degree:
                raise RLClusterWorkloadError(
                    f"{role} mesh degree {degree} must divide Qwen {name}={value}"
                )
    if spec.rollout_micro_batch_size % len(topology.role_to_device_indices["rollout"]):
        raise RLClusterWorkloadError("rollout micro-batch must divide rollout mesh degree")
    if spec.train_micro_batch_size % len(topology.role_to_device_indices["actor"]):
        raise RLClusterWorkloadError("train micro-batch must divide actor mesh degree")
