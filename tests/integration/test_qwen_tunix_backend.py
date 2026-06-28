"""Optional real-Qwen integration smoke; skipped when local weights are absent."""

from __future__ import annotations

from pathlib import Path

import pytest

from tunix_craftext.env.prompts import ActionCatalog, RenderedPrompt
from tunix_craftext.models.llm import LlmRequest
from tunix_craftext.models.tunix_adapter import QwenTunixBackend

ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "artifacts" / "models" / "qwen25-05b-instruct"


@pytest.mark.integration
@pytest.mark.skipif(not SNAPSHOT.is_dir(), reason="download the local Qwen snapshot")
def test_real_qwen_tunix_backend_generates_completion_with_provenance() -> None:
    """One actual Tunix sampling call yields raw text and measured latency."""
    backend = QwenTunixBackend(SNAPSHOT, cache_size=256)
    response = backend.complete(
        LlmRequest(
            RenderedPrompt("Reply only: <action>DO</action>", ActionCatalog(("DO",)), "smoke"),
            max_new_tokens=4,
        )
    )

    assert response.backend == "tunix-single-device"
    assert response.model == "Qwen/Qwen2.5-0.5B-Instruct"
    assert response.raw_text
    assert response.latency_ms is not None and response.latency_ms > 0
    assert response.token_ids is not None and len(response.token_ids) > 0
    assert response.prompt_token_ids is not None and len(response.prompt_token_ids) > 0
    assert response.token_logprobs is not None
    assert len(response.token_ids) == len(response.token_logprobs)


@pytest.mark.integration
@pytest.mark.skipif(not SNAPSHOT.is_dir(), reason="download the local Qwen snapshot")
def test_real_qwen_tunix_backend_completes_two_prompts_in_one_batch() -> None:
    """Pinned Tunix sampler accepts an ordered prompt batch with per-row provenance."""
    backend = QwenTunixBackend(SNAPSHOT, cache_size=256)
    requests = (
        LlmRequest(
            RenderedPrompt("Reply only: <action>DO</action>", ActionCatalog(("DO",)), "smoke"),
            max_new_tokens=4,
        ),
        LlmRequest(
            RenderedPrompt("Reply only: <action>DO</action>", ActionCatalog(("DO",)), "smoke"),
            max_new_tokens=4,
        ),
    )

    responses = backend.complete_batch(requests)

    assert len(responses) == len(requests)
    assert all(response.raw_text for response in responses)
    assert all(response.token_ids for response in responses)
    assert all(response.prompt_token_ids for response in responses)
    assert all(response.token_logprobs is not None for response in responses)


@pytest.mark.integration
@pytest.mark.skipif(not SNAPSHOT.is_dir(), reason="download the local Qwen snapshot")
def test_real_qwen_hidden_state_feature_bridge_has_batch_token_feature_axes() -> None:
    """Pinned Qwen exposes final hidden states without inventing a critic interface."""
    backend = QwenTunixBackend(SNAPSHOT, cache_size=256)
    hidden_states = backend.hidden_states(
        LlmRequest(
            RenderedPrompt("Reply only: <action>DO</action>", ActionCatalog(("DO",)), "smoke")
        )
    )

    assert hidden_states.ndim == 3
    assert hidden_states.shape[0] == 1
    assert hidden_states.shape[1] > 0
    assert hidden_states.shape[2] == 896
