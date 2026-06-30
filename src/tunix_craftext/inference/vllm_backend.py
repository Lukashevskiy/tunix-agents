"""Optional vLLM rollout-generation adapter.

This module deliberately imports vLLM lazily.  CPU/unit tests and documentation
must be able to load the project without installing a Linux/GPU inference stack.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from time import perf_counter

from ..models.llm import LlmResponse
from .contracts import EngineProfile, GenerationBatch, GenerationResult, InferenceBackendError


@dataclass
class VllmInferenceEngine:
    """Synchronous vLLM engine implementing the project inference contract."""

    profile: EngineProfile
    _llm: object | None = None

    @classmethod
    def from_profile(cls, profile: EngineProfile) -> VllmInferenceEngine:
        """Create a vLLM engine from an explicit profile.

        :raises InferenceBackendError: If vLLM is not installed or the profile is invalid.
        """
        if profile.backend != "vllm-offload":
            raise InferenceBackendError("VllmInferenceEngine requires backend='vllm-offload'")
        try:
            from vllm import LLM  # type: ignore[import-not-found]
        except ImportError as error:
            raise InferenceBackendError(
                "vLLM is not installed. Install the optional inference stack on the "
                "target Linux/GPU runner before using backend='vllm-offload'."
            ) from error
        kwargs: dict[str, object] = {
            "model": profile.model,
            "tensor_parallel_size": profile.tensor_parallel_size,
        }
        if profile.max_model_len is not None:
            kwargs["max_model_len"] = profile.max_model_len
        if profile.dtype is not None:
            kwargs["dtype"] = profile.dtype
        return cls(profile=profile, _llm=LLM(**kwargs))

    def generate(self, batch: GenerationBatch) -> GenerationResult:
        """Generate an ordered batch with vLLM and normalize it to `LlmResponse`."""
        if self._llm is None:
            raise InferenceBackendError("vLLM engine is not initialized; call from_profile()")
        try:
            from vllm import SamplingParams  # type: ignore[import-not-found]
        except ImportError as error:
            raise InferenceBackendError("vLLM SamplingParams is unavailable") from error
        stop = tuple(
            dict.fromkeys(stop for request in batch.requests for stop in request.stop_sequences)
        )
        sampling_params = SamplingParams(
            max_tokens=batch.max_new_tokens,
            temperature=batch.temperature,
            stop=list(stop),
            logprobs=1,
        )
        prompts = [request.prompt.text for request in batch.requests]
        started = perf_counter()
        outputs = self._llm.generate(prompts, sampling_params=sampling_params)  # type: ignore[attr-defined]
        latency_ms = (perf_counter() - started) * 1000.0
        if len(outputs) != len(batch.requests):
            raise InferenceBackendError("vLLM changed batch cardinality")
        responses = tuple(
            _response_from_vllm_output(output, self.profile, latency_ms=latency_ms)
            for output in outputs
        )
        return GenerationResult(
            self.profile,
            responses,
            group_id=batch.group_id,
            policy_version=batch.policy_version,
        )

    async def generate_async(self, batch: GenerationBatch) -> GenerationResult:
        """Generate through the same payload contract from an async orchestrator."""
        return await asyncio.to_thread(self.generate, batch)


def _response_from_vllm_output(
    output: object, profile: EngineProfile, *, latency_ms: float
) -> LlmResponse:
    choices = getattr(output, "outputs", None)
    if not choices:
        raise InferenceBackendError("vLLM output did not contain generated choices")
    choice = choices[0]
    raw_text = str(getattr(choice, "text", ""))
    token_ids = tuple(int(token) for token in getattr(choice, "token_ids", ()) or ())
    token_logprobs = _extract_logprobs(getattr(choice, "logprobs", None))
    prompt_token_ids = tuple(int(token) for token in getattr(output, "prompt_token_ids", ()) or ())
    return LlmResponse(
        raw_text=raw_text,
        backend=profile.backend,
        model=profile.model,
        latency_ms=latency_ms,
        token_logprobs=token_logprobs,
        token_ids=token_ids or None,
        prompt_token_ids=prompt_token_ids or None,
    )


def _extract_logprobs(raw_logprobs: object) -> tuple[float, ...] | None:
    if raw_logprobs is None:
        return None
    values: list[float] = []
    for item in raw_logprobs if isinstance(raw_logprobs, list) else []:
        if isinstance(item, dict) and item:
            first = next(iter(item.values()))
            values.append(float(getattr(first, "logprob", first)))
        elif isinstance(item, (int, float)):
            values.append(float(item))
        else:  # pragma: no cover - defensive for third-party objects
            logprob = getattr(item, "logprob", None)
            if logprob is not None:
                values.append(float(logprob))
    return tuple(values) if values else None
