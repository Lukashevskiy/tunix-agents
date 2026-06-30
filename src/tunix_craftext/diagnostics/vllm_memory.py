"""Estimate vLLM rollout memory pressure from a generation YAML config."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..inference.config import GenerationPipelineConfig, load_generation_pipeline_config

BYTES_PER_GIB = 1024**3
DTYPE_BYTES = {
    "bfloat16": 2,
    "float16": 2,
    "half": 2,
    "float32": 4,
    "float": 4,
    "auto": 2,
}


@dataclass(frozen=True, slots=True)
class CudaMemorySnapshot:
    """Current CUDA memory state for one device."""

    free_gib: float
    total_gib: float
    device: str


@dataclass(frozen=True, slots=True)
class VllmMemoryEstimate:
    """JSON-safe vLLM memory estimate for a generation config."""

    config_path: str
    model: str
    device: str
    dtype: str
    gpu_memory_utilization: float
    requested_gib: float
    free_gib: float | None
    total_gib: float | None
    available_after_reservation_gib: float | None
    fits_current_free_memory: bool | None
    max_model_len: int | None
    max_num_seqs: int | None
    max_num_batched_tokens: int | None
    snapshot_weights_gib: float | None
    estimated_kv_cache_gib: float | None
    estimated_model_plus_kv_gib: float | None
    safety_margin_gib: float
    notes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dict."""
        return asdict(self)


def estimate_vllm_memory_from_config(
    config_path: Path,
    *,
    device_index: int = 0,
    memory_provider: Callable[[int], CudaMemorySnapshot | None] | None = None,
    safety_margin_gib: float = 1.0,
) -> VllmMemoryEstimate:
    """Estimate whether a vLLM rollout engine fits current free GPU memory.

    :param config_path: Strict generation YAML path.
    :param device_index: CUDA device index for the runtime memory snapshot.
    :param memory_provider: Optional provider used by tests or custom launchers.
    :param safety_margin_gib: Free-memory margin kept outside vLLM reservation.
    :returns: Conservative startup estimate and explanatory notes.
    """
    config = load_generation_pipeline_config(config_path)
    snapshot = (
        memory_provider(device_index)
        if memory_provider is not None
        else torch_cuda_memory_snapshot(device_index)
    )
    return estimate_vllm_memory(
        config,
        config_path=config_path,
        memory=snapshot,
        safety_margin_gib=safety_margin_gib,
    )


def estimate_vllm_memory(
    config: GenerationPipelineConfig,
    *,
    config_path: Path,
    memory: CudaMemorySnapshot | None,
    safety_margin_gib: float = 1.0,
) -> VllmMemoryEstimate:
    """Estimate vLLM memory from an already parsed generation config."""
    metadata = config.profile.metadata
    utilization = _gpu_memory_utilization(config)
    total_gib = None if memory is None else memory.total_gib
    free_gib = None if memory is None else memory.free_gib
    requested_gib = 0.0 if total_gib is None else total_gib * utilization
    available_after_reservation = (
        None if free_gib is None else free_gib - requested_gib - safety_margin_gib
    )
    model_path = Path(config.profile.model).expanduser()
    weights_gib = _snapshot_weight_files_gib(model_path)
    model_config = _load_model_config(model_path)
    kv_cache_gib = _estimate_kv_cache_gib(
        model_config=model_config,
        dtype=str(config.profile.dtype or "auto"),
        max_model_len=config.profile.max_model_len,
        max_num_seqs=_optional_positive_int(metadata.get("max_num_seqs")),
    )
    estimated_model_plus_kv = None
    if weights_gib is not None or kv_cache_gib is not None:
        estimated_model_plus_kv = (weights_gib or 0.0) + (kv_cache_gib or 0.0)
    notes = _notes(
        memory=memory,
        model_path=model_path,
        weights_gib=weights_gib,
        kv_cache_gib=kv_cache_gib,
        requested_gib=requested_gib,
        safety_margin_gib=safety_margin_gib,
        free_gib=free_gib,
        estimated_model_plus_kv=estimated_model_plus_kv,
    )
    return VllmMemoryEstimate(
        config_path=str(config_path),
        model=config.profile.model,
        device="unknown" if memory is None else memory.device,
        dtype=str(config.profile.dtype or "auto"),
        gpu_memory_utilization=utilization,
        requested_gib=requested_gib,
        free_gib=free_gib,
        total_gib=total_gib,
        available_after_reservation_gib=available_after_reservation,
        fits_current_free_memory=(
            None if free_gib is None else requested_gib + safety_margin_gib <= free_gib
        ),
        max_model_len=config.profile.max_model_len,
        max_num_seqs=_optional_positive_int(metadata.get("max_num_seqs")),
        max_num_batched_tokens=_optional_positive_int(metadata.get("max_num_batched_tokens")),
        snapshot_weights_gib=weights_gib,
        estimated_kv_cache_gib=kv_cache_gib,
        estimated_model_plus_kv_gib=estimated_model_plus_kv,
        safety_margin_gib=safety_margin_gib,
        notes=notes,
    )


def torch_cuda_memory_snapshot(device_index: int = 0) -> CudaMemorySnapshot | None:
    """Return current free/total CUDA memory through torch, or ``None`` on CPU-only hosts."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info(device_index)
    return CudaMemorySnapshot(
        free_gib=free_bytes / BYTES_PER_GIB,
        total_gib=total_bytes / BYTES_PER_GIB,
        device=torch.cuda.get_device_name(device_index),
    )


def _gpu_memory_utilization(config: GenerationPipelineConfig) -> float:
    metadata_value = config.profile.metadata.get("gpu_memory_utilization")
    if isinstance(metadata_value, (int, float)) and not isinstance(metadata_value, bool):
        value = float(metadata_value)
    else:
        value = config.tunix.vllm_hbm_utilization
    if not 0.0 < value <= 1.0:
        raise ValueError("gpu_memory_utilization must be in (0, 1]")
    return value


def _snapshot_weight_files_gib(model_path: Path) -> float | None:
    if not model_path.exists() or not model_path.is_dir():
        return None
    weight_files = tuple(model_path.glob("*.safetensors")) + tuple(model_path.glob("*.bin"))
    if not weight_files:
        return None
    return sum(path.stat().st_size for path in weight_files) / BYTES_PER_GIB


def _load_model_config(model_path: Path) -> Mapping[str, Any] | None:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        return None
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, Mapping) else None


def _estimate_kv_cache_gib(
    *,
    model_config: Mapping[str, Any] | None,
    dtype: str,
    max_model_len: int | None,
    max_num_seqs: int | None,
) -> float | None:
    if model_config is None or max_model_len is None or max_num_seqs is None:
        return None
    hidden_size = _optional_positive_int(model_config.get("hidden_size"))
    layers = _optional_positive_int(model_config.get("num_hidden_layers"))
    heads = _optional_positive_int(model_config.get("num_attention_heads"))
    kv_heads = _optional_positive_int(model_config.get("num_key_value_heads")) or heads
    if hidden_size is None or layers is None or heads is None or kv_heads is None:
        return None
    head_dim = hidden_size // heads
    bytes_per_value = DTYPE_BYTES.get(dtype.lower(), 2)
    kv_bytes = layers * max_num_seqs * max_model_len * 2 * kv_heads * head_dim * bytes_per_value
    return kv_bytes / BYTES_PER_GIB


def _optional_positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _notes(
    *,
    memory: CudaMemorySnapshot | None,
    model_path: Path,
    weights_gib: float | None,
    kv_cache_gib: float | None,
    requested_gib: float,
    safety_margin_gib: float,
    free_gib: float | None,
    estimated_model_plus_kv: float | None,
) -> tuple[str, ...]:
    notes: list[str] = []
    if memory is None:
        notes.append("torch CUDA memory snapshot is unavailable; fit verdict is unknown")
    if not model_path.exists():
        notes.append("local model snapshot is missing; weight and KV estimates are partial")
    elif weights_gib is None:
        notes.append("model snapshot has no .safetensors/.bin files to sum")
    if kv_cache_gib is None:
        notes.append(
            "KV-cache estimate unavailable; need config.json, max_model_len and max_num_seqs"
        )
    if free_gib is not None and requested_gib + safety_margin_gib > free_gib:
        notes.append("vLLM reservation exceeds current free memory plus safety margin")
    if (
        free_gib is not None
        and estimated_model_plus_kv is not None
        and estimated_model_plus_kv + safety_margin_gib > free_gib
    ):
        notes.append(
            "approximate model+KV footprint exceeds current free memory plus safety margin"
        )
    notes.append("exact block allocation is owned by vLLM; this is a preflight estimate")
    return tuple(notes)
