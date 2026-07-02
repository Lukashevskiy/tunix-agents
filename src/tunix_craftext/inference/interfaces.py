"""Unified sync/async LLM generation interface.

This module names the strategy boundary used by rollout orchestrators.  It
adapts the existing project ``InferenceEngine`` contracts without forcing
callers to care whether a backend is natively sync or async.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .contracts import (
    AsyncInferenceEngine,
    GenerationBatch,
    GenerationResult,
    InferenceEngine,
    as_async_engine,
)


class LlmBackend(Protocol):
    """Strategy protocol for generation backends used by rollout collectors."""

    def generate_sync(self, mega_batch: GenerationBatch) -> GenerationResult:
        """Generate one ordered batch synchronously."""
        ...

    async def generate_async(self, mega_batch: GenerationBatch) -> GenerationResult:
        """Generate one ordered batch asynchronously."""
        ...


@dataclass(frozen=True)
class InferenceEngineBackend:
    """Expose existing sync/async inference engines through ``LlmBackend``."""

    sync_engine: InferenceEngine | None = None
    async_engine: AsyncInferenceEngine | None = None

    def __post_init__(self) -> None:
        """Require at least one real generation strategy."""
        if self.sync_engine is None and self.async_engine is None:
            raise ValueError("InferenceEngineBackend requires sync_engine or async_engine")

    def generate_sync(self, mega_batch: GenerationBatch) -> GenerationResult:
        """Generate through the sync engine when one is available."""
        if self.sync_engine is None:
            raise RuntimeError("backend does not provide synchronous generation")
        return self.sync_engine.generate(mega_batch)

    async def generate_async(self, mega_batch: GenerationBatch) -> GenerationResult:
        """Generate through native async or adapt the sync engine."""
        if self.async_engine is not None:
            return await self.async_engine.generate_async(mega_batch)
        assert self.sync_engine is not None
        return await as_async_engine(self.sync_engine).generate_async(mega_batch)
