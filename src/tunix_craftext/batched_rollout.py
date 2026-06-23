"""Host-batched LLM decisions followed by a parallel JAX environment step."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import jax
import jax.numpy as jnp

from .adapters import CrafTextAdapter, EnvironmentStep
from .llm import BatchLlmBackend, LlmRequest, LlmResponse
from .prompts import ActionCatalog, PromptContext, PromptRenderer, RenderedPrompt
from .text_policy import DecodedAction, DecodeMetrics, decode_action, decode_action_outcome


@dataclass(frozen=True)
class BatchedTextDecision:
    """Evidence and transition for one batch of parallel text-environment decisions."""

    prompts: tuple[RenderedPrompt, ...]
    responses: tuple[LlmResponse, ...]
    actions: tuple[DecodedAction, ...]
    metrics: tuple[DecodeMetrics, ...]
    fallback_used: jax.Array
    transition: EnvironmentStep[object, object]


def _item_at(tree: object, index: int) -> object:
    """Select one leading-batch item from an arbitrary JAX state PyTree."""
    return jax.tree.map(lambda leaf: leaf[index], tree)


def collect_batched_text_decision(
    adapter: CrafTextAdapter[object, object, object],
    renderer: PromptRenderer[object],
    backend: BatchLlmBackend,
    *,
    states: object,
    action_masks: jax.Array,
    actions: ActionCatalog,
    keys: jax.Array,
    goal: str,
    max_new_tokens: int,
    dialog: Sequence[tuple[str, ...]] | None = None,
    invalid_action: Literal["error", "fallback"] = "error",
    fallback_action_id: int | None = None,
) -> BatchedTextDecision:
    """Render, complete, decode and step an ordered environment batch.

    Prompt rendering and decoding are intentionally host-side because they use
    text. The model boundary is one ``complete_batch`` call; transition stepping
    is one ``jax.vmap(adapter.step)`` call. This is the synchronous precursor to
    the same transport owned by Tunix ``RLCluster.ROLLOUT``.
    """
    if action_masks.ndim != 2 or action_masks.shape[1] != adapter.action_count:
        raise ValueError("action_masks must have shape [B, adapter.action_count]")
    batch_size = action_masks.shape[0]
    if keys.shape != (batch_size, 2):
        raise ValueError("keys must have shape [B, 2]")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if invalid_action == "fallback" and (
        fallback_action_id is None or not 0 <= fallback_action_id < adapter.action_count
    ):
        raise ValueError("fallback_action_id must be inside the action catalog")
    dialogs = tuple(() for _ in range(batch_size)) if dialog is None else tuple(dialog)
    if len(dialogs) != batch_size:
        raise ValueError("dialog must contain one tuple per batch item")
    prompts = tuple(
        renderer.render(PromptContext(goal, _item_at(states, index), actions, dialogs[index]))
        for index in range(batch_size)
    )
    responses = backend.complete_batch(
        tuple(LlmRequest(prompt, max_new_tokens=max_new_tokens) for prompt in prompts)
    )
    if len(responses) != batch_size:
        raise ValueError("batch backend response cardinality must equal batch size")
    decisions: list[DecodedAction] = []
    metrics: list[DecodeMetrics] = []
    fallback: list[bool] = []
    for prompt, response in zip(prompts, responses, strict=True):
        decision, outcome = decode_action_outcome(prompt, response.raw_text)
        if decision is None:
            if invalid_action == "error":
                decode_action(prompt, response.raw_text)
                raise AssertionError("decode_action must raise for an invalid outcome")
            assert fallback_action_id is not None
            decision = DecodedAction(
                fallback_action_id,
                actions.labels[fallback_action_id],
                response.raw_text,
            )
            fallback.append(True)
        else:
            fallback.append(False)
        decisions.append(decision)
        metrics.append(outcome)
    action_ids = jnp.asarray([decision.action_id for decision in decisions], dtype=jnp.int32)
    transition = jax.vmap(adapter.step)(keys, states, action_ids)
    return BatchedTextDecision(
        prompts=prompts,
        responses=responses,
        actions=tuple(decisions),
        metrics=tuple(metrics),
        fallback_used=jnp.asarray(fallback, dtype=bool),
        transition=transition,
    )
