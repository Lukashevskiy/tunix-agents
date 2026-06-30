"""Optional vLLM adapter contract tests without installing vLLM."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from tunix_craftext.env.prompts import ActionCatalog, RenderedPrompt
from tunix_craftext.inference import EngineProfile, InferenceBackendError, VllmInferenceEngine
from tunix_craftext.inference.contracts import GenerationBatch
from tunix_craftext.models.llm import LlmRequest


def test_vllm_engine_requires_vllm_backend_profile() -> None:
    with pytest.raises(InferenceBackendError, match="backend='vllm-offload'"):
        VllmInferenceEngine.from_profile(EngineProfile("bad", "scripted", "model"))


def test_vllm_engine_fails_cleanly_when_not_initialized() -> None:
    engine = VllmInferenceEngine(EngineProfile("vllm", "vllm-offload", "model"))
    batch = GenerationBatch(
        (LlmRequest(RenderedPrompt("prompt", ActionCatalog(("NOOP",)), "base")),)
    )

    with pytest.raises(InferenceBackendError, match="not initialized"):
        engine.generate(batch)


def test_vllm_engine_explains_torchvision_binary_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("vllm")

    def __getattr__(name: str) -> object:
        if name == "LLM":
            raise RuntimeError("operator torchvision::nms does not exist")
        raise AttributeError(name)

    module.__getattr__ = __getattr__  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", module)

    with pytest.raises(InferenceBackendError, match="torchvision::nms"):
        VllmInferenceEngine.from_profile(EngineProfile("vllm", "vllm-offload", "model"))


def test_vllm_engine_rejects_missing_local_snapshot(tmp_path: Path) -> None:
    missing_snapshot = tmp_path / "missing-qwen"

    with pytest.raises(InferenceBackendError, match="Local vLLM model snapshot is missing"):
        VllmInferenceEngine.from_profile(
            EngineProfile("vllm", "vllm-offload", str(missing_snapshot))
        )


def test_vllm_engine_explains_engine_core_start_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("vllm")

    class BrokenLLM:
        def __init__(self, **kwargs: object) -> None:
            raise RuntimeError(
                "Engine core initialization failed. See root cause above. "
                "Failed core proc(s): {'EngineCore': 1}"
            )

    module.LLM = BrokenLLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", module)

    with pytest.raises(InferenceBackendError, match="make accelerator-stack"):
        VllmInferenceEngine.from_profile(
            EngineProfile(
                "vllm",
                "vllm-offload",
                "Qwen/Qwen2.5-0.5B-Instruct",
                dtype="bfloat16",
                max_model_len=1024,
            )
        )
