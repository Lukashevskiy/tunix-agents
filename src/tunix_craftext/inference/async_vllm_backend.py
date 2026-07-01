"""Native AsyncLLMEngine adapter for experimental continuous vLLM rollout."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import perf_counter
from typing import Any, cast
from uuid import uuid4

from ..models.llm import LlmRequest
from .contracts import EngineProfile, GenerationBatch, GenerationResult, InferenceBackendError
from .vllm_backend import (
    _configure_vllm_environment,
    _raise_vllm_engine_start_error,
    _raise_vllm_runtime_import_error,
    _response_from_vllm_output,
    _validate_model_snapshot,
    _vllm_kwargs_from_profile_metadata,
)


@dataclass
class AsyncVllmInferenceEngine:
    """Native vLLM async engine using AsyncLLMEngine and async generators."""

    profile: EngineProfile
    _engine: object

    @classmethod
    def from_profile(cls, profile: EngineProfile) -> AsyncVllmInferenceEngine:
        """Create a native async vLLM engine from an explicit profile."""
        if profile.backend != "vllm-offload":
            raise InferenceBackendError("AsyncVllmInferenceEngine requires backend='vllm-offload'")
        _validate_model_snapshot(profile.model)
        _configure_vllm_environment(profile)
        try:
            async_engine_class, engine_args_class = _import_async_vllm_classes()
        except ImportError as error:
            raise InferenceBackendError(
                "vLLM AsyncLLMEngine is not installed. Install the optional vLLM stack "
                "on the target Linux/GPU runner before using native async rollout."
            ) from error
        except RuntimeError as error:
            _raise_vllm_runtime_import_error(error)
        kwargs: dict[str, object] = {
            "model": profile.model,
            "tensor_parallel_size": profile.tensor_parallel_size,
        }
        if profile.max_model_len is not None:
            kwargs["max_model_len"] = profile.max_model_len
        if profile.dtype is not None:
            kwargs["dtype"] = profile.dtype
        kwargs.update(_vllm_kwargs_from_profile_metadata(profile))
        try:
            engine_args = engine_args_class(**kwargs)
            engine = async_engine_class.from_engine_args(engine_args)
        except RuntimeError as error:
            _raise_vllm_engine_start_error(error, profile)
        return cls(profile=profile, _engine=engine)

    async def generate_async(self, batch: GenerationBatch) -> GenerationResult:
        """Generate a batch through native vLLM async request streams."""
        semaphore = asyncio.Semaphore(_async_request_concurrency(self.profile, len(batch.requests)))

        async def run_one(index: int, request: LlmRequest) -> tuple[int, object, float]:
            async with semaphore:
                started = perf_counter()
                output = await self._generate_one(request, batch.group_id, index)
                return index, output, (perf_counter() - started) * 1000.0

        outputs = await asyncio.gather(
            *(run_one(index, request) for index, request in enumerate(batch.requests))
        )
        ordered = tuple(output for _, output, _ in sorted(outputs, key=lambda item: item[0]))
        latencies = tuple(latency for _, _, latency in sorted(outputs, key=lambda item: item[0]))
        responses = tuple(
            _response_from_vllm_output(output, self.profile, latency_ms=latency)
            for output, latency in zip(ordered, latencies, strict=True)
        )
        return GenerationResult(
            self.profile,
            responses,
            group_id=batch.group_id,
            policy_version=batch.policy_version,
        )

    async def _generate_one(
        self, request: LlmRequest, group_id: str | None, index: int
    ) -> object:
        """Consume vLLM's async generator and return the final RequestOutput."""
        try:
            from vllm import SamplingParams  # type: ignore[import-not-found]
        except ImportError as error:
            raise InferenceBackendError("vLLM SamplingParams is unavailable") from error
        except RuntimeError as error:
            _raise_vllm_runtime_import_error(error)
        sampling_params = SamplingParams(
            max_tokens=request.max_new_tokens,
            temperature=request.temperature,
            stop=list(request.stop_sequences),
            logprobs=1,
        )
        request_id = f"{self.profile.name}:{group_id or 'batch'}:{index}:{uuid4().hex}"
        final_output = None
        engine = cast(Any, self._engine)
        async for output in engine.generate(
            request.prompt.text,
            sampling_params,
            request_id,
        ):
            final_output = output
        if final_output is None:
            raise InferenceBackendError("vLLM async generator produced no outputs")
        return final_output


def _import_async_vllm_classes() -> tuple[Any, Any]:
    """Import vLLM AsyncLLMEngine/AsyncEngineArgs across supported vLLM layouts."""
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs  # type: ignore[import-not-found]
        from vllm.engine.async_llm_engine import (  # type: ignore[import-not-found]
            AsyncLLMEngine,
        )

        return AsyncLLMEngine, AsyncEngineArgs
    except ImportError:
        from vllm import AsyncEngineArgs, AsyncLLMEngine  # type: ignore[import-not-found]

        return AsyncLLMEngine, AsyncEngineArgs


def _async_request_concurrency(profile: EngineProfile, request_count: int) -> int:
    """Return per-batch native async request concurrency."""
    raw = profile.metadata.get("async_request_concurrency")
    if raw is None:
        raw = profile.metadata.get("max_in_flight")
    if raw is None:
        return max(1, request_count)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise InferenceBackendError(
            "engine.metadata.async_request_concurrency must be a positive integer"
        )
    return min(raw, max(1, request_count))
