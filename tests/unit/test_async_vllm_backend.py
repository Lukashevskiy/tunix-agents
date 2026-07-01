"""Native AsyncLLMEngine adapter tests without installing vLLM."""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from tunix_craftext.env.prompts import ActionCatalog, RenderedPrompt
from tunix_craftext.inference import (
    AsyncVllmInferenceEngine,
    EngineProfile,
    GenerationBatch,
    InferenceBackendError,
    VllmInferenceEngine,
    build_async_inference_engine,
)
from tunix_craftext.models.llm import LlmRequest


def test_sync_vllm_engine_rejects_fake_async_wrapper() -> None:
    engine = VllmInferenceEngine(EngineProfile("vllm", "vllm-offload", "model"))

    with pytest.raises(InferenceBackendError, match="AsyncVllmInferenceEngine"):
        asyncio.run(engine.generate_async(_batch(("first",))))


def test_async_vllm_engine_uses_native_async_generator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_args: dict[str, object] = {}
    seen_requests: list[str] = []
    _install_fake_async_vllm(monkeypatch, seen_args=seen_args, seen_requests=seen_requests)

    engine = AsyncVllmInferenceEngine.from_profile(
        EngineProfile(
            "async-vllm",
            "vllm-offload",
            "Qwen/Qwen2.5-0.5B-Instruct",
            max_model_len=2048,
            dtype="bfloat16",
            mode="async",
            metadata={
                "gpu_memory_utilization": 0.25,
                "max_num_batched_tokens": 4096,
                "max_num_seqs": 32,
                "async_request_concurrency": 1,
            },
        )
    )

    result = asyncio.run(engine.generate_async(_batch(("first", "second"))))

    assert seen_args["gpu_memory_utilization"] == 0.25
    assert seen_args["max_num_batched_tokens"] == 4096
    assert seen_args["max_model_len"] == 2048
    assert seen_requests == ["first", "second"]
    assert [response.raw_text for response in result.responses] == [
        "<action>first</action>",
        "<action>second</action>",
    ]


def test_async_vllm_factory_builds_native_vllm_async_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_async_vllm(monkeypatch, seen_args={}, seen_requests=[])

    engine = build_async_inference_engine(
        EngineProfile("async-vllm", "vllm-offload", "Qwen/Qwen2.5-0.5B-Instruct")
    )

    assert isinstance(engine, AsyncVllmInferenceEngine)


def _batch(labels: tuple[str, ...]) -> GenerationBatch:
    catalog = ActionCatalog(labels)
    requests = tuple(
        LlmRequest(RenderedPrompt(label, catalog, "base"), max_new_tokens=4)
        for label in labels
    )
    return GenerationBatch(requests, group_id="group", policy_version=7)


def _install_fake_async_vllm(
    monkeypatch: pytest.MonkeyPatch,
    *,
    seen_args: dict[str, object],
    seen_requests: list[str],
) -> None:
    vllm = types.ModuleType("vllm")
    engine_pkg = types.ModuleType("vllm.engine")
    arg_utils = types.ModuleType("vllm.engine.arg_utils")
    async_llm_engine = types.ModuleType("vllm.engine.async_llm_engine")

    class SamplingParams:
        def __init__(self, **kwargs: object) -> None:
            self.kwargs = kwargs

    class AsyncEngineArgs:
        def __init__(self, **kwargs: object) -> None:
            seen_args.update(kwargs)

    class FakeChoice:
        def __init__(self, text: str) -> None:
            self.text = text
            self.token_ids = (1, 2)
            self.logprobs = None

    class FakeOutput:
        def __init__(self, text: str) -> None:
            self.outputs = [FakeChoice(text)]
            self.prompt_token_ids = (10, 11)

    class AsyncLLMEngine:
        @classmethod
        def from_engine_args(cls, args: object) -> AsyncLLMEngine:
            return cls()

        async def generate(
            self, prompt: str, sampling_params: SamplingParams, request_id: str
        ):
            seen_requests.append(prompt)
            yield FakeOutput(f"<action>{prompt}</action>")

    vllm.SamplingParams = SamplingParams  # type: ignore[attr-defined]
    arg_utils.AsyncEngineArgs = AsyncEngineArgs  # type: ignore[attr-defined]
    async_llm_engine.AsyncLLMEngine = AsyncLLMEngine  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.engine", engine_pkg)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", arg_utils)
    monkeypatch.setitem(sys.modules, "vllm.engine.async_llm_engine", async_llm_engine)
