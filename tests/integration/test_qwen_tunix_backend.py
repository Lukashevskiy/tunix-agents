"""Optional real-Qwen integration smoke; skipped when local weights are absent."""

from __future__ import annotations

from pathlib import Path

import pytest

from tunix_craftext.llm import LlmRequest
from tunix_craftext.prompts import ActionCatalog, RenderedPrompt
from tunix_craftext.tunix_adapter import QwenTunixBackend


ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT = ROOT / "artifacts" / "models" / "qwen25-05b-instruct"


@pytest.mark.integration
@pytest.mark.skipif(not SNAPSHOT.is_dir(), reason="download the local Qwen snapshot")
def test_real_qwen_tunix_backend_generates_completion_with_provenance() -> None:
    """One actual Tunix sampling call yields raw text and measured latency."""
    backend = QwenTunixBackend(SNAPSHOT, cache_size=128)
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
