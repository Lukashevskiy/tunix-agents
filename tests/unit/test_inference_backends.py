"""Inference backend registry and adapter tests."""

from __future__ import annotations

import pytest

from tunix_craftext.env.prompts import ActionCatalog, RenderedPrompt
from tunix_craftext.inference import (
    EngineProfile,
    GenerationBatch,
    InferenceBackendError,
    SglangInferenceEngine,
    VanillaInferenceEngine,
    build_inference_engine,
)
from tunix_craftext.models.llm import LlmRequest, ScriptedLlmBackend


def _batch() -> GenerationBatch:
    request = LlmRequest(RenderedPrompt("prompt", ActionCatalog(("NOOP",)), "base"))
    return GenerationBatch((request,), group_id="g", policy_version=4)


def test_vanilla_inference_engine_wraps_existing_batch_backend() -> None:
    profile = EngineProfile("scripted", "scripted", "fixture")
    engine = VanillaInferenceEngine(profile, ScriptedLlmBackend("<action>NOOP</action>"))

    result = engine.generate(_batch())

    assert result.profile is profile
    assert result.group_id == "g"
    assert result.policy_version == 4
    assert result.responses[0].raw_text == "<action>NOOP</action>"


def test_registry_builds_vanilla_engine_from_supplied_backend() -> None:
    profile = EngineProfile("local", "tunix-single-device", "qwen")

    engine = build_inference_engine(
        profile,
        vanilla_backends={"local": ScriptedLlmBackend("<action>NOOP</action>")},
    )

    assert engine.generate(_batch()).responses[0].backend == "scripted"


def test_registry_rejects_missing_vanilla_backend_instance() -> None:
    with pytest.raises(InferenceBackendError, match="requires a supplied BatchLlmBackend"):
        build_inference_engine(EngineProfile("missing", "vanilla", "model"))


def test_sglang_backend_is_explicit_but_not_silent() -> None:
    engine = SglangInferenceEngine.from_profile(EngineProfile("sg", "sglang", "model"))

    with pytest.raises(InferenceBackendError, match="not implemented"):
        engine.generate(_batch())
