"""Strict inference profile compiler for existing Tunix rollout configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from .contracts import EngineProfile, InferenceBackendError

TunixRolloutEngineName = Literal["vanilla", "vllm", "sglang_jax"]


@dataclass(frozen=True)
class TunixGenerationContract:
    """Project-level generation contract that compiles to Tunix `RolloutConfig`.

    The class intentionally mirrors only stable knobs we own semantically.  More
    backend-specific options pass through explicit dictionaries so new Tunix
    fields can be adopted without changing rollout payload contracts.
    """

    engine: TunixRolloutEngineName
    max_prompt_length: int
    max_tokens_to_generate: int
    temperature: float = 0.0
    kv_cache_size: int = 1024
    return_logprobs: bool = True
    tensor_parallel_size: int = -1
    data_parallel_size: int = -1
    expert_parallel_size: int = 1
    vllm_server_mode: bool = False
    vllm_async_scheduling: bool = False
    vllm_hbm_utilization: float = 0.2
    vllm_model_version: str = ""
    vllm_init_with_random_weights: bool = True
    vllm_max_num_batched_tokens: int | None = None
    vllm_max_num_seqs: int | None = None
    vllm_kwargs: dict[str, object] = field(default_factory=dict)
    vllm_sampling_kwargs: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate generation configuration before compiling to Tunix."""
        if self.engine not in {"vanilla", "vllm", "sglang_jax"}:
            raise InferenceBackendError(f"unsupported Tunix rollout engine: {self.engine}")
        for name, value in (
            ("max_prompt_length", self.max_prompt_length),
            ("max_tokens_to_generate", self.max_tokens_to_generate),
            ("kv_cache_size", self.kv_cache_size),
            ("expert_parallel_size", self.expert_parallel_size),
        ):
            if value <= 0:
                raise InferenceBackendError(f"{name} must be positive")
        if self.tensor_parallel_size == 0 or self.data_parallel_size == 0:
            raise InferenceBackendError("parallel sizes must be positive or -1 for Tunix default")
        if not 0.0 <= self.vllm_hbm_utilization <= 1.0:
            raise InferenceBackendError("vllm_hbm_utilization must be in [0, 1]")

    def engine_profile(self, *, name: str, model: str, dtype: str | None = None) -> EngineProfile:
        """Return the normalized project inference profile for this Tunix contract."""
        backend = "vllm-offload" if self.engine == "vllm" else f"tunix-{self.engine}"
        return EngineProfile(
            name=name,
            backend=backend,
            model=model,
            tensor_parallel_size=max(self.tensor_parallel_size, 1),
            max_model_len=self.max_prompt_length + self.max_tokens_to_generate,
            dtype=dtype,
            mode="async" if self.vllm_async_scheduling or self.vllm_server_mode else "sync",
            metadata={
                "tunix_rollout_engine": self.engine,
                "return_logprobs": self.return_logprobs,
            },
        )

    def to_tunix_rollout_kwargs(self) -> dict[str, object]:
        """Compile to keyword arguments accepted by Tunix `RolloutConfig`."""
        return {
            "max_tokens_to_generate": self.max_tokens_to_generate,
            "temperature": self.temperature,
            "max_prompt_length": self.max_prompt_length,
            "kv_cache_size": self.kv_cache_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "data_parallel_size": self.data_parallel_size,
            "expert_parallel_size": self.expert_parallel_size,
            "return_logprobs": self.return_logprobs,
            "rollout_vllm_server_mode": self.vllm_server_mode,
            "rollout_vllm_model_version": self.vllm_model_version,
            "rollout_vllm_hbm_utilization": self.vllm_hbm_utilization,
            "rollout_vllm_init_with_random_weights": self.vllm_init_with_random_weights,
            "rollout_vllm_async_scheduling": self.vllm_async_scheduling,
            "rollout_vllm_max_num_batched_tokens": self.vllm_max_num_batched_tokens,
            "rollout_vllm_max_num_seqs": self.vllm_max_num_seqs,
            "rollout_vllm_kwargs": dict(self.vllm_kwargs),
            "rollout_vllm_sampling_kwargs": dict(self.vllm_sampling_kwargs),
        }

    def to_tunix_rollout_config(self) -> object:
        """Instantiate Tunix `RolloutConfig` lazily when the `tunix` extra exists."""
        try:
            from tunix.rl.rollout.base_rollout import RolloutConfig  # type: ignore[import-untyped]
        except ImportError as error:
            raise InferenceBackendError(
                "install tunix-craftext[tunix] to build RolloutConfig"
            ) from error
        return RolloutConfig(**self.to_tunix_rollout_kwargs())
