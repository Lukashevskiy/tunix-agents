"""LLM actor boundary tests independent from heavyweight model loading."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

from tunix_craftext.env.prompts import ActionCatalog, RenderedPrompt
from tunix_craftext.models.llm import LlmRequest
from tunix_craftext.models.llm_actor import DeterministicLlmActor
from tunix_craftext.models.profile import load_model_profile

ROOT = Path(__file__).resolve().parents[2]


def _actor() -> DeterministicLlmActor:
    return DeterministicLlmActor(
        load_model_profile(ROOT / "configs/models/gemma3_270m_instruction.yaml"),
        action_text="<action>LEFT</action>",
        token_bucket_count=64,
    )


def test_deterministic_llm_actor_generates_batch_with_backbone_profile() -> None:
    actor = _actor()
    requests = (
        LlmRequest(RenderedPrompt("first prompt", ActionCatalog(("LEFT",)), "base")),
        LlmRequest(RenderedPrompt("second prompt", ActionCatalog(("LEFT",)), "base")),
    )

    responses = actor.generate_batch(requests)

    assert len(responses) == 2
    assert all(response.model == "google/gemma-3-270m-it" for response in responses)
    assert all(response.token_ids for response in responses)
    assert all(response.token_logprobs for response in responses)


def test_deterministic_llm_actor_scores_tokens_with_prompt_conditioning() -> None:
    actor = _actor()

    scores = actor.score_tokens(
        prompt_token_ids=jnp.array([[1, 2, 0], [3, 0, 0]], dtype=jnp.int32),
        prompt_token_mask=jnp.array([[True, True, False], [True, False, False]]),
        token_ids=jnp.array([[5, 6], [7, 8]], dtype=jnp.int32),
        token_mask=jnp.array([[True, True], [True, False]]),
    )

    assert scores.token_logprobs.shape == (2, 2)
    assert scores.values.shape == (2, 2)
    assert scores.entropy.shape == (2, 2)
    assert bool(jnp.all(jnp.isfinite(scores.token_logprobs)))
    assert float(scores.entropy[1, 1]) == 0.0


def test_deterministic_llm_actor_rejects_misaligned_prompt_and_token_batches() -> None:
    actor = _actor()

    with pytest.raises(ValueError, match="same B axis"):
        actor.score_tokens(
            prompt_token_ids=jnp.ones((2, 3), dtype=jnp.int32),
            prompt_token_mask=jnp.ones((2, 3), dtype=bool),
            token_ids=jnp.ones((1, 2), dtype=jnp.int32),
            token_mask=jnp.ones((1, 2), dtype=bool),
        )
