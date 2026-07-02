"""Tests for vLLM/Tunix anti-corruption tensor adapter."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from tunix_craftext.env.prompts import ActionCatalog, RenderedPrompt
from tunix_craftext.inference import (
    EngineProfile,
    GenerationBatch,
    GenerationResult,
    InferenceBackendError,
    InferenceEngineBackend,
    VllmToTunixAdapter,
)
from tunix_craftext.models.llm import LlmRequest, LlmResponse


def test_vllm_to_tunix_adapter_pads_normalized_responses() -> None:
    result = GenerationResult(
        EngineProfile("test", "vllm-offload", "model"),
        (
            LlmResponse(
                "a",
                "vllm",
                "model",
                token_ids=(11, 12),
                token_logprobs=(-0.1, -0.2),
            ),
            LlmResponse(
                "b",
                "vllm",
                "model",
                token_ids=(13,),
                token_logprobs=(-0.3,),
            ),
        ),
    )

    tensors = VllmToTunixAdapter(pad_token_id=-1).convert_generation_result(result)

    np.testing.assert_array_equal(tensors.generation_tokens, np.array([[11, 12], [13, -1]]))
    np.testing.assert_allclose(tensors.actor_log_probs, np.array([[-0.1, -0.2], [-0.3, 0.0]]))
    np.testing.assert_array_equal(tensors.generation_mask, np.array([[True, True], [True, False]]))
    assert tensors.raw_text == ("a", "b")


def test_vllm_to_tunix_adapter_rejects_missing_logprobs_by_default() -> None:
    with pytest.raises(InferenceBackendError, match="missing token logprobs"):
        VllmToTunixAdapter().convert_responses(
            (LlmResponse("a", "vllm", "model", token_ids=(1,)),)
        )


def test_vllm_to_tunix_adapter_can_fill_missing_logprobs_when_explicit() -> None:
    tensors = VllmToTunixAdapter(
        allow_missing_logprobs=True,
        missing_logprob=-99.0,
    ).convert_responses((LlmResponse("a", "vllm", "model", token_ids=(1, 2)),))

    np.testing.assert_allclose(tensors.actor_log_probs, np.array([[-99.0, -99.0]]))


def test_vllm_to_tunix_adapter_rejects_overlong_generation() -> None:
    with pytest.raises(InferenceBackendError, match="exceeds max_length"):
        VllmToTunixAdapter(max_length=1).convert_responses(
            (
                LlmResponse(
                    "a",
                    "vllm",
                    "model",
                    token_ids=(1, 2),
                    token_logprobs=(-0.1, -0.2),
                ),
            )
        )


def test_vllm_to_tunix_adapter_extracts_raw_vllm_outputs() -> None:
    output = SimpleNamespace(
        outputs=(
            SimpleNamespace(
                text="<action>NOOP</action>",
                token_ids=(7, 8),
                logprobs=({7: SimpleNamespace(logprob=-0.7)}, {8: -0.8}),
            ),
        )
    )

    tensors = VllmToTunixAdapter().convert((output,))

    np.testing.assert_array_equal(tensors.generation_tokens, np.array([[7, 8]]))
    np.testing.assert_allclose(tensors.actor_log_probs, np.array([[-0.7, -0.8]]))
    np.testing.assert_array_equal(tensors.generation_mask, np.array([[True, True]]))


class FakeSyncEngine:
    profile = EngineProfile("sync", "scripted", "model")

    def generate(self, batch: GenerationBatch) -> GenerationResult:
        return GenerationResult(
            self.profile,
            tuple(
                LlmResponse(request.prompt.text, "scripted", "model")
                for request in batch.requests
            ),
        )


def _batch() -> GenerationBatch:
    return GenerationBatch(
        (LlmRequest(RenderedPrompt("hello", ActionCatalog(("NOOP",)), "base")),)
    )


def test_inference_engine_backend_wraps_sync_and_async_paths() -> None:
    backend = InferenceEngineBackend(sync_engine=FakeSyncEngine())

    sync_result = backend.generate_sync(_batch())
    async_result = asyncio.run(backend.generate_async(_batch()))

    assert sync_result.responses[0].raw_text == "hello"
    assert async_result.responses[0].raw_text == "hello"


def test_inference_engine_backend_rejects_empty_strategy() -> None:
    with pytest.raises(ValueError, match="requires sync_engine or async_engine"):
        InferenceEngineBackend()
