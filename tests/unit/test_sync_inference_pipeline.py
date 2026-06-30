"""Synchronous generation collection tests."""

from __future__ import annotations

from tunix_craftext.env.prompts import ActionCatalog, RenderedPrompt
from tunix_craftext.inference import (
    EngineProfile,
    GenerationBatch,
    GenerationResult,
    collect_generation_results_sync,
)
from tunix_craftext.models.llm import LlmRequest, LlmResponse


class RecordingSyncEngine:
    def __init__(self) -> None:
        self.profile = EngineProfile("sync-fake", "scripted", "fake")
        self.seen: list[str | None] = []

    def generate(self, batch: GenerationBatch) -> GenerationResult:
        self.seen.append(batch.group_id)
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
    return GenerationBatch((request,), group_id=group_id, policy_version=2)


def test_sync_generation_results_preserve_submission_order_and_policy_version() -> None:
    engine = RecordingSyncEngine()

    records = collect_generation_results_sync(engine, (_batch("a"), _batch("b")))

    assert engine.seen == ["a", "b"]
    assert [record.index for record in records] == [0, 1]
    assert [record.result.group_id for record in records] == ["a", "b"]
    assert [record.result.policy_version for record in records] == [2, 2]
