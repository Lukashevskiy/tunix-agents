"""Static mesh/model/batch checks performed before Qwen weights enter memory."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .rlcluster_workload import AgenticGrpoWorkloadSpec, RLClusterWorkloadError
from .topology import TunixTopology

RolloutBackend = Literal[
    "vanilla-jax-sharded",
    "single-device-jax",
    "vllm-offload",
    "sglang-jax",
    "scripted",
    "evidence",
]


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
    *,
    rollout_backend: RolloutBackend = "vanilla-jax-sharded",
    model_family: str = "qwen2",
    allow_known_broken_sharded_qwen_rollout: bool = False,
) -> None:
    """Reject impossible role/backend/model contracts before model load.

    This preflight is intentionally stricter than plain tensor divisibility.
    The current Tunix Qwen vanilla sampler can fail during the embedding gather
    when a Qwen actor/reference is loaded on an ``fsdp,tp`` mesh, even if both
    symbolic axes have degree one.  Evidence/scripted checks and future
    generation backends remain allowed; the known-broken sharded Qwen rollout is
    stopped here with an actionable error before weights enter memory.
    """
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
    _validate_rollout_backend_contract(
        topology,
        rollout_backend=rollout_backend,
        model_family=model_family,
        allow_known_broken_sharded_qwen_rollout=allow_known_broken_sharded_qwen_rollout,
    )


def _validate_rollout_backend_contract(
    topology: TunixTopology,
    *,
    rollout_backend: RolloutBackend,
    model_family: str,
    allow_known_broken_sharded_qwen_rollout: bool,
) -> None:
    if rollout_backend not in {
        "vanilla-jax-sharded",
        "single-device-jax",
        "vllm-offload",
        "sglang-jax",
        "scripted",
        "evidence",
    }:
        raise RLClusterWorkloadError(f"unsupported rollout_backend: {rollout_backend}")
    if allow_known_broken_sharded_qwen_rollout:
        return
    family = model_family.lower().replace("-", "").replace("_", "")
    if family not in {"qwen", "qwen2", "qwen25", "qwen25instruct"}:
        return
    if rollout_backend != "vanilla-jax-sharded":
        return
    axes = tuple(part.strip() for part in topology.axis_name.split(","))
    if "tp" not in axes:
        return
    raise RLClusterWorkloadError(
        "Qwen vanilla-jax-sharded rollout on an fsdp/tp Tunix mesh is disabled: "
        "the current Tunix Qwen sampler can fail in the embedding gather before "
        "generation starts. Use rollout_backend='scripted'/'evidence' for checks, "
        "or implement the planned 'single-device-jax' or 'vllm-offload' rollout boundary "
        "before running real Agentic GRPO."
    )
