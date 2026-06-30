"""Backend-neutral rollout-generation contracts inspired by JAX-Toolbox boundaries.

The key architectural rule is that generation is an inference workload, not an
implicit side effect of the trainer mesh.  A trainer can own actor/reference/
critic state while rollout generation goes through a separate engine with an
explicit profile, batch contract and provenance.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol

from ..models.llm import BatchLlmBackend, LlmRequest, LlmResponse

GenerationMode = Literal["sync", "async"]


class InferenceBackendError(RuntimeError):
    """Raised when an optional inference backend is unavailable or misconfigured."""


@dataclass(frozen=True)
class EngineProfile:
    """Static description of one inference engine instance.

    :param name: Stable engine name, for example ``local-qwen-vllm``.
    :param backend: Backend family: ``tunix-single-device``, ``vllm-offload``, etc.
    :param model: Model id or local snapshot path consumed by the engine.
    :param tensor_parallel_size: Real tensor-parallel degree, not a symbolic mesh axis.
    :param max_model_len: Maximum total prompt+generation length accepted by the engine.
    :param dtype: Storage/compute dtype label surfaced in evidence.
    :param mode: Whether the engine is called through a synchronous or asynchronous contract.
    :param policy_version: Optional actor policy version whose weights produced generations.
    :param metadata: Additional JSON-safe profile metadata.
    """

    name: str
    backend: str
    model: str
    tensor_parallel_size: int = 1
    max_model_len: int | None = None
    dtype: str | None = None
    mode: GenerationMode = "sync"
    policy_version: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject invalid static engine configuration before backend allocation."""
        if not self.name.strip() or not self.backend.strip() or not self.model.strip():
            raise InferenceBackendError("engine profile name, backend and model must be non-empty")
        if self.tensor_parallel_size <= 0:
            raise InferenceBackendError("tensor_parallel_size must be positive")
        if self.max_model_len is not None and self.max_model_len <= 0:
            raise InferenceBackendError("max_model_len must be positive when provided")
        if self.policy_version is not None and self.policy_version < 0:
            raise InferenceBackendError("policy_version must be non-negative when provided")


@dataclass(frozen=True)
class GenerationBatch:
    """One ordered static batch sent to a rollout inference engine."""

    requests: tuple[LlmRequest, ...]
    seed: int = 0
    group_id: str | None = None
    policy_version: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate cardinality and compatible decoding knobs."""
        if not self.requests:
            raise InferenceBackendError("generation batch must contain at least one request")
        max_new_tokens = {request.max_new_tokens for request in self.requests}
        temperatures = {request.temperature for request in self.requests}
        if len(max_new_tokens) != 1:
            raise InferenceBackendError("all requests in one generation batch need max_new_tokens")
        if len(temperatures) != 1:
            raise InferenceBackendError("all requests in one generation batch need one temperature")
        if self.policy_version is not None and self.policy_version < 0:
            raise InferenceBackendError("policy_version must be non-negative when provided")

    @property
    def max_new_tokens(self) -> int:
        """Return the shared maximum generation length."""
        return self.requests[0].max_new_tokens

    @property
    def temperature(self) -> float:
        """Return the shared sampling temperature."""
        return self.requests[0].temperature


@dataclass(frozen=True)
class GenerationResult:
    """Ordered inference result with backend profile provenance."""

    profile: EngineProfile
    responses: tuple[LlmResponse, ...]
    group_id: str | None = None
    policy_version: int | None = None

    def __post_init__(self) -> None:
        """Ensure callers cannot lose batch cardinality by accident."""
        if not self.responses:
            raise InferenceBackendError("generation result must contain at least one response")


class InferenceEngine(Protocol):
    """Synchronous rollout-generation boundary used by GRPO/PPO collectors."""

    @property
    def profile(self) -> EngineProfile:
        """Return static engine provenance."""
        ...

    def generate(self, batch: GenerationBatch) -> GenerationResult:
        """Generate one ordered batch without depending on trainer mesh internals."""
        ...


class AsyncInferenceEngine(Protocol):
    """Asynchronous rollout-generation boundary with the same payload contract."""

    @property
    def profile(self) -> EngineProfile:
        """Return static engine provenance."""
        ...

    async def generate_async(self, batch: GenerationBatch) -> GenerationResult:
        """Generate one ordered batch asynchronously."""
        ...


@dataclass(frozen=True)
class SyncToAsyncInferenceEngine:
    """Adapter for using any synchronous engine from an async rollout orchestrator."""

    engine: InferenceEngine

    @property
    def profile(self) -> EngineProfile:
        """Return wrapped engine provenance with async call mode surfaced."""
        return self.engine.profile

    async def generate_async(self, batch: GenerationBatch) -> GenerationResult:
        """Run sync generation in a worker thread without changing the payload schema."""
        return await asyncio.to_thread(self.engine.generate, batch)


@dataclass(frozen=True)
class RequestsLlmBackend:
    """Adapter exposing an ``InferenceEngine`` through the existing LLM backend API."""

    engine: InferenceEngine

    def complete(self, request: LlmRequest) -> LlmResponse:
        """Generate a single request through the wrapped inference engine."""
        return self.complete_batch((request,))[0]

    def complete_batch(self, requests: Sequence[LlmRequest]) -> tuple[LlmResponse, ...]:
        """Generate an ordered request batch through the wrapped inference engine."""
        result = self.engine.generate(GenerationBatch(tuple(requests)))
        if len(result.responses) != len(requests):
            raise InferenceBackendError("inference engine changed batch cardinality")
        return result.responses


def as_batch_llm_backend(engine: InferenceEngine) -> BatchLlmBackend:
    """Return an existing rollout-compatible backend wrapper for an inference engine."""
    return RequestsLlmBackend(engine)


def as_async_engine(engine: InferenceEngine | AsyncInferenceEngine) -> AsyncInferenceEngine:
    """Normalize sync or async generation implementations to the async contract."""
    if hasattr(engine, "generate_async"):
        return engine  # type: ignore[return-value]
    return SyncToAsyncInferenceEngine(engine)  # type: ignore[arg-type]
