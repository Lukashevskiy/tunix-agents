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


def test_vllm_engine_sets_jax_safe_worker_start_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("vllm")
    seen_env: dict[str, str | None] = {}

    class FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            import os

            seen_env["method"] = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")

    module.LLM = FakeLLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.delenv("VLLM_WORKER_MULTIPROC_METHOD", raising=False)

    VllmInferenceEngine.from_profile(
        EngineProfile("vllm", "vllm-offload", "Qwen/Qwen2.5-0.5B-Instruct")
    )

    assert seen_env == {"method": "spawn"}


def test_vllm_engine_preserves_user_worker_start_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("vllm")
    seen_env: dict[str, str | None] = {}

    class FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            import os

            seen_env["method"] = os.environ.get("VLLM_WORKER_MULTIPROC_METHOD")

    module.LLM = FakeLLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "forkserver")

    VllmInferenceEngine.from_profile(
        EngineProfile(
            "vllm",
            "vllm-offload",
            "Qwen/Qwen2.5-0.5B-Instruct",
            metadata={"multiprocessing_method": "spawn"},
        )
    )

    assert seen_env == {"method": "forkserver"}


def test_vllm_engine_can_configure_vllm_v1_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("vllm")
    seen_env: dict[str, str | None] = {}

    class FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            import os

            seen_env["v1"] = os.environ.get("VLLM_USE_V1")

    module.LLM = FakeLLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", module)
    monkeypatch.delenv("VLLM_USE_V1", raising=False)

    VllmInferenceEngine.from_profile(
        EngineProfile(
            "vllm",
            "vllm-offload",
            "Qwen/Qwen2.5-0.5B-Instruct",
            metadata={"vllm_use_v1": False},
        )
    )

    assert seen_env == {"v1": "0"}


def test_vllm_engine_passes_memory_and_batch_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("vllm")
    seen_kwargs: dict[str, object] = {}

    class FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            seen_kwargs.update(kwargs)

    module.LLM = FakeLLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", module)

    VllmInferenceEngine.from_profile(
        EngineProfile(
            "vllm",
            "vllm-offload",
            "Qwen/Qwen2.5-0.5B-Instruct",
            metadata={
                "gpu_memory_utilization": 0.35,
                "max_num_batched_tokens": 4096,
                "max_num_seqs": 32,
                "disable_log_stats": True,
            },
        )
    )

    assert seen_kwargs["gpu_memory_utilization"] == 0.35
    assert seen_kwargs["max_num_batched_tokens"] == 4096
    assert seen_kwargs["max_num_seqs"] == 32
    assert seen_kwargs["disable_log_stats"] is True


def test_vllm_engine_rejects_invalid_memory_utilization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("vllm")

    class FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            raise AssertionError("LLM must not be constructed for invalid metadata")

    module.LLM = FakeLLM  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vllm", module)

    with pytest.raises(InferenceBackendError, match="gpu_memory_utilization"):
        VllmInferenceEngine.from_profile(
            EngineProfile(
                "vllm",
                "vllm-offload",
                "Qwen/Qwen2.5-0.5B-Instruct",
                metadata={"gpu_memory_utilization": 1.5},
            )
        )
