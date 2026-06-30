"""Async generation collection tests."""

from __future__ import annotations

import asyncio

import pytest

from tunix_craftext.env.prompts import ActionCatalog, RenderedPrompt
from tunix_craftext.inference import (
    EngineProfile,
    GenerationBatch,
    GenerationResult,
    collect_generation_results,
)
from tunix_craftext.models.llm import LlmRequest, LlmResponse


class DelayedAsyncEngine:
    def __init__(self) -> None:
        self.profile = EngineProfile("async-fake", "scripted", "fake", mode="async")
        self.started: list[str | None] = []

    async def generate_async(self, batch: GenerationBatch) -> GenerationResult:
        self.started.append(batch.group_id)
        delay = 0.02 if batch.group_id == "slow" else 0.0
        await asyncio.sleep(delay)
        return GenerationResult(
            self.profile,
            (
                LlmResponse(
                    f"<action>{batch.requests[0].prompt.actions.labels[0]}</action>",
                    self.profile.backend,
                    self.profile.model,
                ),
            ),
            group_id=batch.group_id,
            policy_version=batch.policy_version,
        )


def _batch(group_id: str) -> GenerationBatch:
    request = LlmRequest(RenderedPrompt(group_id, ActionCatalog(("NOOP",)), "base"))
    return GenerationBatch((request,), group_id=group_id, policy_version=1)


def test_async_generation_results_preserve_submission_order() -> None:
    engine = DelayedAsyncEngine()

    records = asyncio.run(
        collect_generation_results(
            engine,
            (_batch("slow"), _batch("fast")),
            max_in_flight=2,
        )
    )

    assert [record.batch.group_id for record in records] == ["slow", "fast"]
    assert [record.result.group_id for record in records] == ["slow", "fast"]
    assert [record.result.responses[0].raw_text for record in records] == [
        "<action>NOOP</action>",
        "<action>NOOP</action>",
    ]


def test_async_generation_results_reject_invalid_concurrency() -> None:
    with pytest.raises(ValueError, match="max_in_flight"):
        asyncio.run(
            collect_generation_results(
                DelayedAsyncEngine(),
                (_batch("a"),),
                max_in_flight=0,
            )
        )
