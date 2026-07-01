"""Inference-engine boundary tests for rollout generation."""

from __future__ import annotations

import asyncio

import pytest

from tunix_craftext.env.prompts import ActionCatalog, RenderedPrompt
from tunix_craftext.inference import (
    EngineProfile,
    GenerationBatch,
    GenerationResult,
    InferenceBackendError,
    RequestsLlmBackend,
    TunixGenerationContract,
    as_async_engine,
    local_vllm_rollout_contract,
)
from tunix_craftext.models.llm import LlmRequest, LlmResponse


class FakeInferenceEngine:
    def __init__(self) -> None:
        self.profile = EngineProfile("fake", "scripted", "fake-model")
        self.last_batch: GenerationBatch | None = None

    def generate(self, batch: GenerationBatch) -> GenerationResult:
        self.last_batch = batch
        return GenerationResult(
            self.profile,
            tuple(
                LlmResponse(
                    f"<action>{request.prompt.actions.labels[0]}</action>",
                    self.profile.backend,
                    self.profile.model,
                    latency_ms=1.0,
                )
                for request in batch.requests
            ),
        )


def _request(text: str, *, max_new_tokens: int = 4, temperature: float = 0.0) -> LlmRequest:
    return LlmRequest(
        RenderedPrompt(text, ActionCatalog(("NOOP", "LEFT")), "base"),
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )


def test_generation_batch_rejects_mixed_static_decoding_knobs() -> None:
    with pytest.raises(InferenceBackendError, match="max_new_tokens"):
        GenerationBatch((_request("a", max_new_tokens=4), _request("b", max_new_tokens=8)))

    with pytest.raises(InferenceBackendError, match="temperature"):
        GenerationBatch((_request("a", temperature=0.0), _request("b", temperature=1.0)))


def test_generation_batch_carries_policy_version_and_group_id() -> None:
    batch = GenerationBatch((_request("a"),), group_id="task-7", policy_version=3)

    assert batch.group_id == "task-7"
    assert batch.policy_version == 3


def test_inference_engine_adapter_preserves_existing_batch_llm_contract() -> None:
    engine = FakeInferenceEngine()
    backend = RequestsLlmBackend(engine)

    responses = backend.complete_batch((_request("first"), _request("second")))

    assert engine.last_batch is not None
    assert [request.prompt.text for request in engine.last_batch.requests] == ["first", "second"]
    assert [response.raw_text for response in responses] == [
        "<action>NOOP</action>",
        "<action>NOOP</action>",
    ]


def test_engine_profile_rejects_symbolic_zero_parallelism() -> None:
    with pytest.raises(InferenceBackendError, match="tensor_parallel_size"):
        EngineProfile("bad", "vllm-offload", "model", tensor_parallel_size=0)


def test_sync_engine_can_be_normalized_to_async_contract() -> None:
    engine = FakeInferenceEngine()
    async_engine = as_async_engine(engine)

    result = asyncio.run(async_engine.generate_async(GenerationBatch((_request("async"),))))

    assert result.responses[0].raw_text == "<action>NOOP</action>"


def test_tunix_generation_contract_compiles_vllm_rollout_kwargs() -> None:
    contract = TunixGenerationContract(
        engine="vllm",
        max_prompt_length=128,
        max_tokens_to_generate=16,
        tensor_parallel_size=1,
        vllm_server_mode=True,
        vllm_async_scheduling=True,
        vllm_model_version="qwen2.5-0.5b",
        vllm_max_num_batched_tokens=4096,
    )

    kwargs = contract.to_tunix_rollout_kwargs()
    profile = contract.engine_profile(name="rollout", model="qwen")

    assert kwargs["rollout_vllm_server_mode"] is True
    assert kwargs["rollout_vllm_async_scheduling"] is True
    assert kwargs["rollout_vllm_model_version"] == "qwen2.5-0.5b"
    assert kwargs["rollout_vllm_max_num_batched_tokens"] == 4096
    assert profile.backend == "vllm-offload"
    assert profile.mode == "async"


def test_local_vllm_rollout_contract_replaces_tunix_alias_with_snapshot_path() -> None:
    """Runtime vLLM rollout must receive a local snapshot, not a Tunix model alias."""
    contract = TunixGenerationContract(
        engine="vllm",
        max_prompt_length=128,
        max_tokens_to_generate=16,
        tensor_parallel_size=1,
        vllm_server_mode=False,
        vllm_async_scheduling=True,
        vllm_model_version="qwen2.5-0.5b",
        vllm_init_with_random_weights=True,
    )

    runtime_contract = local_vllm_rollout_contract(
        contract,
        "/models/qwen25-05b-instruct",
        server_mode=True,
        async_scheduling=False,
    )
    kwargs = runtime_contract.to_tunix_rollout_kwargs()

    assert runtime_contract.engine == "vllm"
    assert kwargs["rollout_vllm_model_version"] == "/models/qwen25-05b-instruct"
    assert kwargs["rollout_vllm_init_with_random_weights"] is False
    assert kwargs["rollout_vllm_server_mode"] is True
    assert kwargs["rollout_vllm_async_scheduling"] is False
