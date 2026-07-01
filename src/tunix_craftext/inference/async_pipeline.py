"""Async generation collection utilities shared by vLLM and future rollout engines."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from time import perf_counter

from .contracts import AsyncInferenceEngine, GenerationBatch, InferenceEngine, as_async_engine
from .sync_pipeline import GenerationRecord, GenerationTiming, ProfiledGenerationRecord

AsyncGenerationRecord = GenerationRecord


async def collect_generation_results(
    engine: AsyncInferenceEngine,
    batches: Sequence[GenerationBatch],
    *,
    max_in_flight: int = 1,
) -> tuple[GenerationRecord, ...]:
    """Collect generation batches concurrently while preserving submission order.

    :param engine: Async inference engine used for all batches.
    :param batches: Ordered generation batches to execute.
    :param max_in_flight: Bounded concurrency; ``1`` is a strict async wrapper over sync flow.
    :returns: Records sorted by the original batch order.
    """
    if max_in_flight <= 0:
        raise ValueError("max_in_flight must be positive")
    semaphore = asyncio.Semaphore(max_in_flight)

    async def run_one(index: int, batch: GenerationBatch) -> GenerationRecord:
        async with semaphore:
            result = await engine.generate_async(batch)
        return GenerationRecord(index, batch, result)

    records = await asyncio.gather(
        *(run_one(index, batch) for index, batch in enumerate(tuple(batches)))
    )
    return tuple(sorted(records, key=lambda record: record.index))


async def collect_generation_results_profiled(
    engine: AsyncInferenceEngine,
    batches: Sequence[GenerationBatch],
    *,
    max_in_flight: int = 1,
) -> tuple[ProfiledGenerationRecord, ...]:
    """Collect generation batches concurrently with queue/backend timing evidence."""
    if max_in_flight <= 0:
        raise ValueError("max_in_flight must be positive")
    semaphore = asyncio.Semaphore(max_in_flight)

    async def run_one(index: int, batch: GenerationBatch) -> ProfiledGenerationRecord:
        submitted = perf_counter()
        async with semaphore:
            acquired = perf_counter()
            result = await engine.generate_async(batch)
            finished = perf_counter()
        response_latency = tuple(
            float(response.latency_ms)
            for response in result.responses
            if response.latency_ms is not None
        )
        return ProfiledGenerationRecord(
            GenerationRecord(index, batch, result),
            GenerationTiming(
                queued_ms=(acquired - submitted) * 1000.0,
                backend_ms=(finished - acquired) * 1000.0,
                total_ms=(finished - submitted) * 1000.0,
                response_latency_ms=response_latency,
            ),
        )

    records = await asyncio.gather(
        *(run_one(index, batch) for index, batch in enumerate(tuple(batches)))
    )
    return tuple(sorted(records, key=lambda record: record.record.index))


async def collect_generation_results_from_sync_engine(
    engine: InferenceEngine | AsyncInferenceEngine,
    batches: Sequence[GenerationBatch],
    *,
    max_in_flight: int = 1,
) -> tuple[GenerationRecord, ...]:
    """Normalize sync/async engines and collect batches with one payload contract."""
    return await collect_generation_results(
        as_async_engine(engine), batches, max_in_flight=max_in_flight
    )
