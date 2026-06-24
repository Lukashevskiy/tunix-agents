"""Host-batched LLM decisions followed by a parallel JAX environment step."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import jax
import jax.numpy as jnp

from .adapters import CrafTextAdapter, EnvironmentReset, EnvironmentStep
from .llm import BatchLlmBackend, LlmRequest, LlmResponse
from .prompts import ActionCatalog, PromptContext, PromptRenderer, RenderedPrompt
from .replay import ReplayArtifact, ReplayStep
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


@dataclass(frozen=True)
class BatchedTextRollout:
    """Fixed-horizon parallel decisions with explicit post-terminal reset masks."""

    initial_reset: EnvironmentReset[object, object]
    decisions: tuple[BatchedTextDecision, ...]
    reset_after_step: tuple[jax.Array, ...]


def _item_at(tree: object, index: int) -> object:
    """Select one leading-batch item from an arbitrary JAX state PyTree."""
    return jax.tree.map(lambda leaf: leaf[index], tree)


def _replace_finished(current: object, reset: object, finished: jax.Array) -> object:
    """Select reset leaves only for finished leading-batch rows."""
    def select(current_leaf: jax.Array, reset_leaf: jax.Array) -> jax.Array:
        shape = (finished.shape[0],) + (1,) * (current_leaf.ndim - 1)
        return jnp.where(finished.reshape(shape), reset_leaf, current_leaf)

    return jax.tree.map(select, current, reset)


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
    for row_index, (prompt, response) in enumerate(zip(prompts, responses, strict=True)):
        decision, outcome = decode_action_outcome(prompt, response.raw_text)
        if decision is not None and not bool(action_masks[row_index, decision.action_id]):
            outcome = DecodeMetrics(masked_action=1)
            decision = None
        if decision is None:
            if invalid_action == "error":
                if outcome.masked_action:
                    raise ValueError("model selected an action masked out by the environment")
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


def collect_batched_text_rollout(
    adapter: CrafTextAdapter[object, object, object],
    renderer: PromptRenderer[object],
    backend: BatchLlmBackend,
    *,
    actions: ActionCatalog,
    batch_size: int,
    horizon: int,
    seed: int,
    goal: str,
    max_new_tokens: int,
    invalid_action: Literal["error", "fallback"] = "error",
    fallback_action_id: int | None = None,
) -> BatchedTextRollout:
    """Collect a fixed-horizon multi-env rollout with explicit done-reset semantics.

    Resets are evaluated for every row and selected only where terminal/truncated;
    this keeps the state PyTree static and avoids Python branching over JAX data.
    Each returned decision still records the pre-reset terminal transition.
    """
    if batch_size <= 0 or horizon <= 0:
        raise ValueError("batch_size and horizon must be positive")
    keys = jax.random.split(jax.random.PRNGKey(seed), batch_size + horizon * 2 * batch_size)
    reset = jax.vmap(adapter.reset)(keys[:batch_size])
    state = reset.state
    action_mask = reset.action_mask
    dialogs: tuple[tuple[str, ...], ...] = tuple(() for _ in range(batch_size))
    decisions: list[BatchedTextDecision] = []
    reset_masks: list[jax.Array] = []
    cursor = batch_size
    for _ in range(horizon):
        step_keys = keys[cursor : cursor + batch_size]
        reset_keys = keys[cursor + batch_size : cursor + 2 * batch_size]
        cursor += 2 * batch_size
        decision = collect_batched_text_decision(
            adapter,
            renderer,
            backend,
            states=state,
            action_masks=action_mask,
            actions=actions,
            keys=step_keys,
            goal=goal,
            max_new_tokens=max_new_tokens,
            dialog=dialogs,
            invalid_action=invalid_action,
            fallback_action_id=fallback_action_id,
        )
        finished = jnp.logical_or(decision.transition.terminated, decision.transition.truncated)
        fresh = jax.vmap(adapter.reset)(reset_keys)
        state = _replace_finished(decision.transition.state, fresh.state, finished)
        action_mask = cast(
            jax.Array,
            _replace_finished(decision.transition.action_mask, fresh.action_mask, finished),
        )
        dialogs = tuple(
            () if bool(finished[index]) else (*dialogs[index], decision.responses[index].raw_text)
            for index in range(batch_size)
        )
        decisions.append(decision)
        reset_masks.append(finished)
    return BatchedTextRollout(reset, tuple(decisions), tuple(reset_masks))


def replays_from_batched_rollout(
    rollout: BatchedTextRollout, *, config_path: str, commit: str, backend: str
) -> tuple[ReplayArtifact, ...]:
    """Split a batch rollout into inspectable per-environment replay artifacts."""
    if not rollout.decisions:
        raise ValueError("rollout must contain at least one decision")
    batch_size = rollout.decisions[0].transition.reward.shape[0]
    artifacts: list[ReplayArtifact] = []
    for environment_index in range(batch_size):
        steps: list[ReplayStep] = []
        for step_index, decision in enumerate(rollout.decisions):
            transition = decision.transition
            response = decision.responses[environment_index]
            action = decision.actions[environment_index]
            metrics = decision.metrics[environment_index]
            steps.append(
                ReplayStep(
                    index=step_index,
                    prompt=decision.prompts[environment_index].text,
                    raw_completion=response.raw_text,
                    action_id=action.action_id,
                    action_label=action.label,
                    reward=float(transition.reward[environment_index]),
                    terminated=bool(transition.terminated[environment_index]),
                    truncated=bool(transition.truncated[environment_index]),
                    action_mask=tuple(
                        bool(value) for value in transition.action_mask[environment_index]
                    ),
                    observation=_item_at(transition.observation, environment_index),
                    invalid_format=metrics.invalid_format,
                    unknown_action=metrics.unknown_action,
                    fallback_used=bool(decision.fallback_used[environment_index]),
                    token_logprobs=response.token_logprobs,
                    token_ids=response.token_ids,
                    prompt_token_ids=response.prompt_token_ids,
                )
            )
        artifacts.append(ReplayArtifact(config_path, commit, backend, tuple(steps)))
    return tuple(artifacts)
