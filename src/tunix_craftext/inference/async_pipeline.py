"""Async generation collection utilities shared by vLLM and future rollout engines."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass

from .contracts import AsyncInferenceEngine, GenerationBatch, GenerationResult, as_async_engine


@dataclass(frozen=True)
class AsyncGenerationRecord:
    """One ordered async generation result with queue index provenance."""

    index: int
    batch: GenerationBatch
    result: GenerationResult


async def collect_generation_results(
    engine: AsyncInferenceEngine,
    batches: Sequence[GenerationBatch],
    *,
    max_in_flight: int = 1,
) -> tuple[AsyncGenerationRecord, ...]:
    """Collect generation batches concurrently while preserving submission order.

    :param engine: Async inference engine used for all batches.
    :param batches: Ordered generation batches to execute.
    :param max_in_flight: Bounded concurrency; ``1`` is a strict async wrapper over sync flow.
    :returns: Records sorted by the original batch order.
    """
    if max_in_flight <= 0:
        raise ValueError("max_in_flight must be positive")
    semaphore = asyncio.Semaphore(max_in_flight)

    async def run_one(index: int, batch: GenerationBatch) -> AsyncGenerationRecord:
        async with semaphore:
            result = await engine.generate_async(batch)
        return AsyncGenerationRecord(index, batch, result)

    records = await asyncio.gather(
        *(run_one(index, batch) for index, batch in enumerate(tuple(batches)))
    )
    return tuple(sorted(records, key=lambda record: record.index))


async def collect_generation_results_from_sync_engine(
    engine: object,
    batches: Sequence[GenerationBatch],
    *,
    max_in_flight: int = 1,
) -> tuple[AsyncGenerationRecord, ...]:
    """Normalize sync/async engines and collect batches with one payload contract."""
    return await collect_generation_results(
        as_async_engine(engine), batches, max_in_flight=max_in_flight
    )
