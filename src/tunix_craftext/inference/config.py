"""Versioned YAML configuration for sync/async rollout generation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .contracts import EngineProfile, GenerationMode, InferenceBackendError
from .tunix_config import TunixGenerationContract, TunixRolloutEngineName


@dataclass(frozen=True)
class AsyncCollectionConfig:
    """Queue/concurrency knobs for async generation collection."""

    max_in_flight: int = 1
    queue_maxsize: int = 1
    submission_timeout_s: float = 0.0

    def __post_init__(self) -> None:
        """Reject impossible async collection settings."""
        if self.max_in_flight <= 0:
            raise InferenceBackendError("async.max_in_flight must be positive")
        if self.queue_maxsize <= 0:
            raise InferenceBackendError("async.queue_maxsize must be positive")
        if self.submission_timeout_s < 0:
            raise InferenceBackendError("async.submission_timeout_s must be non-negative")


@dataclass(frozen=True)
class GenerationPipelineConfig:
    """Complete declarative generation config for rollout collectors and Tunix."""

    schema_version: int
    profile: EngineProfile
    tunix: TunixGenerationContract
    async_collection: AsyncCollectionConfig

    def __post_init__(self) -> None:
        """Validate version and profile/contract consistency."""
        if self.schema_version != 1:
            raise InferenceBackendError(
                f"unsupported generation schema_version: {self.schema_version}"
            )
        expected_model_len = self.tunix.max_prompt_length + self.tunix.max_tokens_to_generate
        if (
            self.profile.max_model_len is not None
            and self.profile.max_model_len < expected_model_len
        ):
            raise InferenceBackendError(
                "engine.max_model_len must fit tunix.max_prompt_length + "
                "tunix.max_tokens_to_generate"
            )


def load_generation_pipeline_config(path: Path) -> GenerationPipelineConfig:
    """Load one strict generation YAML config."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise InferenceBackendError(f"cannot read generation config: {path}") from error
    root = _mapping(raw, "root")
    _keys(root, {"schema_version", "engine", "tunix", "async"}, "root")
    return GenerationPipelineConfig(
        schema_version=_int(root["schema_version"], "schema_version"),
        profile=_engine_profile(_mapping(root["engine"], "engine")),
        tunix=_tunix_contract(_mapping(root["tunix"], "tunix")),
        async_collection=_async_collection(_mapping(root["async"], "async")),
    )


def generation_config_to_manifest(
    config: GenerationPipelineConfig,
    *,
    path: Path,
) -> dict[str, Any]:
    """Serialize generation config provenance for run manifests."""
    return {
        "path": str(path),
        "engine": {
            "name": config.profile.name,
            "backend": config.profile.backend,
            "model": config.profile.model,
            "tensor_parallel_size": config.profile.tensor_parallel_size,
            "max_model_len": config.profile.max_model_len,
            "dtype": config.profile.dtype,
            "mode": config.profile.mode,
            "policy_version": config.profile.policy_version,
            "metadata": dict(config.profile.metadata),
        },
        "tunix": config.tunix.to_tunix_rollout_kwargs(),
        "async": {
            "max_in_flight": config.async_collection.max_in_flight,
            "queue_maxsize": config.async_collection.queue_maxsize,
            "submission_timeout_s": config.async_collection.submission_timeout_s,
        },
    }


def _engine_profile(value: Mapping[str, object]) -> EngineProfile:
    _keys(
        value,
        {
            "name",
            "backend",
            "model",
            "tensor_parallel_size",
            "max_model_len",
            "dtype",
            "mode",
            "policy_version",
            "metadata",
        },
        "engine",
    )
    metadata = _mapping(value["metadata"], "engine.metadata")
    return EngineProfile(
        name=_string(value["name"], "engine.name"),
        backend=_string(value["backend"], "engine.backend"),
        model=_string(value["model"], "engine.model"),
        tensor_parallel_size=_positive_int(
            value["tensor_parallel_size"], "engine.tensor_parallel_size"
        ),
        max_model_len=_positive_int(value["max_model_len"], "engine.max_model_len"),
        dtype=_optional_string(value["dtype"], "engine.dtype"),
        mode=_mode(value["mode"], "engine.mode"),
        policy_version=_optional_non_negative_int(
            value["policy_version"], "engine.policy_version"
        ),
        metadata=dict(metadata),
    )


def _tunix_contract(value: Mapping[str, object]) -> TunixGenerationContract:
    _keys(
        value,
        {
            "engine",
            "max_prompt_length",
            "max_tokens_to_generate",
            "temperature",
            "kv_cache_size",
            "return_logprobs",
            "tensor_parallel_size",
            "data_parallel_size",
            "expert_parallel_size",
            "vllm_server_mode",
            "vllm_async_scheduling",
            "vllm_hbm_utilization",
            "vllm_model_version",
            "vllm_init_with_random_weights",
            "vllm_max_num_batched_tokens",
            "vllm_max_num_seqs",
            "vllm_kwargs",
            "vllm_sampling_kwargs",
        },
        "tunix",
    )
    return TunixGenerationContract(
        engine=_tunix_engine(value["engine"], "tunix.engine"),
        max_prompt_length=_positive_int(value["max_prompt_length"], "tunix.max_prompt_length"),
        max_tokens_to_generate=_positive_int(
            value["max_tokens_to_generate"], "tunix.max_tokens_to_generate"
        ),
        temperature=_float(value["temperature"], "tunix.temperature"),
        kv_cache_size=_positive_int(value["kv_cache_size"], "tunix.kv_cache_size"),
        return_logprobs=_bool(value["return_logprobs"], "tunix.return_logprobs"),
        tensor_parallel_size=_minus_one_or_positive_int(
            value["tensor_parallel_size"], "tunix.tensor_parallel_size"
        ),
        data_parallel_size=_minus_one_or_positive_int(
            value["data_parallel_size"], "tunix.data_parallel_size"
        ),
        expert_parallel_size=_positive_int(
            value["expert_parallel_size"], "tunix.expert_parallel_size"
        ),
        vllm_server_mode=_bool(value["vllm_server_mode"], "tunix.vllm_server_mode"),
        vllm_async_scheduling=_bool(
            value["vllm_async_scheduling"], "tunix.vllm_async_scheduling"
        ),
        vllm_hbm_utilization=_float(
            value["vllm_hbm_utilization"], "tunix.vllm_hbm_utilization"
        ),
        vllm_model_version=_string(value["vllm_model_version"], "tunix.vllm_model_version"),
        vllm_init_with_random_weights=_bool(
            value["vllm_init_with_random_weights"], "tunix.vllm_init_with_random_weights"
        ),
        vllm_max_num_batched_tokens=_optional_positive_int(
            value["vllm_max_num_batched_tokens"], "tunix.vllm_max_num_batched_tokens"
        ),
        vllm_max_num_seqs=_optional_positive_int(
            value["vllm_max_num_seqs"], "tunix.vllm_max_num_seqs"
        ),
        vllm_kwargs=dict(_mapping(value["vllm_kwargs"], "tunix.vllm_kwargs")),
        vllm_sampling_kwargs=dict(
            _mapping(value["vllm_sampling_kwargs"], "tunix.vllm_sampling_kwargs")
        ),
    )


def _async_collection(value: Mapping[str, object]) -> AsyncCollectionConfig:
    _keys(value, {"max_in_flight", "queue_maxsize", "submission_timeout_s"}, "async")
    return AsyncCollectionConfig(
        max_in_flight=_positive_int(value["max_in_flight"], "async.max_in_flight"),
        queue_maxsize=_positive_int(value["queue_maxsize"], "async.queue_maxsize"),
        submission_timeout_s=_float(value["submission_timeout_s"], "async.submission_timeout_s"),
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise InferenceBackendError(f"{name} must be a mapping with string keys")
    return value


def _keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise InferenceBackendError(
            f"{name} keys must be exactly {sorted(expected)}, got {sorted(value)}"
        )


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InferenceBackendError(f"{name} must be a non-empty string")
    return value


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name)


def _int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise InferenceBackendError(f"{name} must be an integer")
    return value


def _positive_int(value: object, name: str) -> int:
    result = _int(value, name)
    if result <= 0:
        raise InferenceBackendError(f"{name} must be positive")
    return result


def _optional_positive_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _positive_int(value, name)


def _optional_non_negative_int(value: object, name: str) -> int | None:
    if value is None:
        return None
    result = _int(value, name)
    if result < 0:
        raise InferenceBackendError(f"{name} must be non-negative")
    return result


def _minus_one_or_positive_int(value: object, name: str) -> int:
    result = _int(value, name)
    if result == -1 or result > 0:
        return result
    raise InferenceBackendError(f"{name} must be -1 or positive")


def _float(value: object, name: str) -> float:
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        raise InferenceBackendError(f"{name} must be a float")
    return float(value)


def _bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise InferenceBackendError(f"{name} must be a boolean")
    return value


def _mode(value: object, name: str) -> GenerationMode:
    text = _string(value, name)
    if text not in {"sync", "async"}:
        raise InferenceBackendError(f"{name} must be 'sync' or 'async'")
    return text  # type: ignore[return-value]


def _tunix_engine(value: object, name: str) -> TunixRolloutEngineName:
    text = _string(value, name)
    if text not in {"vanilla", "vllm", "sglang_jax"}:
        raise InferenceBackendError(f"{name} must be vanilla, vllm or sglang_jax")
    return text  # type: ignore[return-value]
