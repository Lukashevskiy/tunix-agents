"""Host-batched LLM decisions followed by a parallel JAX environment step."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

from ..adapters import CraftaxAdapter, EnvironmentReset, EnvironmentStep
from ..artifacts.replay import ReplayArtifact, ReplayStep
from ..env.prompts import (
    ActionCatalog,
    PromptContext,
    PromptRenderer,
    RenderedPrompt,
    compose_craftext_goal,
)
from ..env.text_policy import DecodedAction, DecodeMetrics, decode_action, decode_action_outcome
from ..models.llm import BatchLlmBackend, LlmRequest, LlmResponse


@dataclass(frozen=True)
class EnvironmentDevicePolicy:
    """Placement policy for batched JAX environment reset/step execution.

    The default leaves JAX free to use its normal device placement. Supplying a
    backend such as ``"cuda"``/``"gpu"`` and ``device_index=0`` explicitly places
    keys, states, action masks and decoded action ids on the selected device
    before vmapped reset/step.  Prompt rendering and action decoding remain
    host-side because they are text/Python boundaries.
    """

    backend: str | None = None
    device_index: int = 0
    jit_reset: bool = True
    jit_step: bool = True
    place_inputs: bool = True
    snapshot_prompt_inputs: bool = True


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


@dataclass(frozen=True)
class BatchedDecisionTiming:
    """Host/model/env phase timings for one batched text decision in milliseconds."""

    prompt_snapshot_ms: float
    prompt_render_ms: float
    llm_batch_ms: float
    action_decode_ms: float
    environment_step_ms: float
    total_ms: float


@dataclass(frozen=True)
class BatchedRolloutStepTiming:
    """End-to-end timing for one rollout step, including resets and dialog bookkeeping."""

    decision: BatchedDecisionTiming
    reset_ms: float
    replace_finished_ms: float
    dialog_update_ms: float
    total_ms: float


@dataclass(frozen=True)
class ProfiledBatchedTextRollout:
    """Batched text rollout together with per-step phase timings.

    This is intentionally opt-in: the regular collector keeps returning the
    compact rollout artifact, while this wrapper exposes enough timing evidence
    to diagnose slow synchronous trajectory collection when vLLM itself looks
    fast and GPU utilization is low.
    """

    rollout: BatchedTextRollout
    timings: tuple[BatchedRolloutStepTiming, ...]

    def phase_totals_ms(self) -> dict[str, float]:
        """Aggregate rollout timing by phase for a quick bottleneck report."""
        totals = {
            "prompt_snapshot_ms": 0.0,
            "prompt_render_ms": 0.0,
            "llm_batch_ms": 0.0,
            "action_decode_ms": 0.0,
            "environment_step_ms": 0.0,
            "reset_ms": 0.0,
            "replace_finished_ms": 0.0,
            "dialog_update_ms": 0.0,
            "total_ms": 0.0,
        }
        for timing in self.timings:
            totals["prompt_snapshot_ms"] += timing.decision.prompt_snapshot_ms
            totals["prompt_render_ms"] += timing.decision.prompt_render_ms
            totals["llm_batch_ms"] += timing.decision.llm_batch_ms
            totals["action_decode_ms"] += timing.decision.action_decode_ms
            totals["environment_step_ms"] += timing.decision.environment_step_ms
            totals["reset_ms"] += timing.reset_ms
            totals["replace_finished_ms"] += timing.replace_finished_ms
            totals["dialog_update_ms"] += timing.dialog_update_ms
            totals["total_ms"] += timing.total_ms
        return totals


def _item_at(tree: object, index: int) -> object:
    """Select one leading-batch item from an arbitrary JAX state PyTree."""
    return jax.tree.map(lambda leaf: leaf[index], tree)


def _replace_finished(current: object, reset: object, finished: jax.Array) -> object:
    """Select reset leaves only for finished leading-batch rows."""

    def select(current_leaf: jax.Array, reset_leaf: jax.Array) -> jax.Array:
        shape = (finished.shape[0],) + (1,) * (current_leaf.ndim - 1)
        return jnp.where(finished.reshape(shape), reset_leaf, current_leaf)

    return jax.tree.map(select, current, reset)


def _resolve_device(policy: EnvironmentDevicePolicy | None) -> jax.Device | None:
    """Resolve an optional environment device placement policy."""
    if policy is None or policy.backend is None:
        return None
    backend = "gpu" if policy.backend == "cuda" else policy.backend
    try:
        devices = jax.devices(backend)
    except RuntimeError as error:
        raise ValueError(
            f"JAX backend is not available for environment: {policy.backend}"
        ) from error
    if not 0 <= policy.device_index < len(devices):
        raise ValueError(
            f"environment device_index {policy.device_index} outside visible {backend} devices"
        )
    return devices[policy.device_index]


def _place_for_environment(value: object, device: jax.Device | None, *, enabled: bool) -> object:
    """Place an arbitrary JAX PyTree on the selected environment device."""
    if device is None or not enabled:
        return value
    return jax.device_put(value, device)


def _host_snapshot_for_prompt(value: object, policy: EnvironmentDevicePolicy | None) -> object:
    """Copy one prompt-visible PyTree snapshot to host when using an explicit device lane."""
    if policy is None or not policy.snapshot_prompt_inputs:
        return value
    return jax.device_get(value)


def _batched_reset(
    adapter: CraftaxAdapter[object, object, object], policy: EnvironmentDevicePolicy | None
):
    """Return a vmapped and optionally jitted reset function."""
    reset = jax.vmap(adapter.reset)
    return jax.jit(reset) if policy is not None and policy.jit_reset else reset


def _batched_step(
    adapter: CraftaxAdapter[object, object, object], policy: EnvironmentDevicePolicy | None
):
    """Return a vmapped and optionally jitted step function."""
    step = jax.vmap(adapter.step)
    return jax.jit(step) if policy is not None and policy.jit_step else step


def collect_batched_text_decision(
    adapter: CraftaxAdapter[object, object, object],
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
    device_policy: EnvironmentDevicePolicy | None = None,
    _inputs_are_placed: bool = False,
    _timing_sink: Callable[[BatchedDecisionTiming], None] | None = None,
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
    timing_enabled = _timing_sink is not None
    total_started_at = perf_counter() if timing_enabled else 0.0
    device = _resolve_device(device_policy)
    place_inputs = True if device_policy is None else device_policy.place_inputs
    should_place_inputs = place_inputs and not _inputs_are_placed
    states = _place_for_environment(states, device, enabled=should_place_inputs)
    action_masks = cast(
        jax.Array, _place_for_environment(action_masks, device, enabled=should_place_inputs)
    )
    keys = cast(jax.Array, _place_for_environment(keys, device, enabled=should_place_inputs))
    snapshot_started_at = perf_counter() if timing_enabled else 0.0
    prompt_states = _host_snapshot_for_prompt(states, device_policy)
    prompt_action_masks = np.asarray(jax.device_get(action_masks))
    prompt_snapshot_ms = (
        (perf_counter() - snapshot_started_at) * 1000.0 if timing_enabled else 0.0
    )
    dialogs = tuple(() for _ in range(batch_size)) if dialog is None else tuple(dialog)
    if len(dialogs) != batch_size:
        raise ValueError("dialog must contain one tuple per batch item")
    render_started_at = perf_counter() if timing_enabled else 0.0
    prompts = tuple(
        renderer.render(
            PromptContext(
                goal
                if context is None
                else compose_craftext_goal(
                    goal,
                    scenario_instruction=context.instruction,
                    world_preset=context.world_preset,
                    text_constraint=context.text_constraint,
                ),
                adapter.prompt_state(state),
                actions,
                dialogs[index],
                safety="" if context is None else context.text_constraint,
                world_preset="" if context is None else context.world_preset,
            )
        )
        for index in range(batch_size)
        for state in (_item_at(prompt_states, index),)
        for context in (
            adapter.episode_context(state) if adapter.has_instruction_context else None,
        )
    )
    prompt_render_ms = (perf_counter() - render_started_at) * 1000.0 if timing_enabled else 0.0
    llm_started_at = perf_counter() if timing_enabled else 0.0
    responses = backend.complete_batch(
        tuple(LlmRequest(prompt, max_new_tokens=max_new_tokens) for prompt in prompts)
    )
    llm_batch_ms = (perf_counter() - llm_started_at) * 1000.0 if timing_enabled else 0.0
    if len(responses) != batch_size:
        raise ValueError("batch backend response cardinality must equal batch size")
    decode_started_at = perf_counter() if timing_enabled else 0.0
    decisions: list[DecodedAction] = []
    metrics: list[DecodeMetrics] = []
    fallback: list[bool] = []
    for row_index, (prompt, response) in enumerate(zip(prompts, responses, strict=True)):
        decision, outcome = decode_action_outcome(prompt, response.raw_text)
        if decision is not None and not bool(prompt_action_masks[row_index, decision.action_id]):
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
    action_ids = cast(jax.Array, _place_for_environment(action_ids, device, enabled=place_inputs))
    action_decode_ms = (perf_counter() - decode_started_at) * 1000.0 if timing_enabled else 0.0
    step_started_at = perf_counter() if timing_enabled else 0.0
    transition = _batched_step(adapter, device_policy)(keys, states, action_ids)
    if timing_enabled:
        transition.reward.block_until_ready()
    environment_step_ms = (perf_counter() - step_started_at) * 1000.0 if timing_enabled else 0.0
    if timing_enabled:
        assert _timing_sink is not None
        _timing_sink(
            BatchedDecisionTiming(
                prompt_snapshot_ms=prompt_snapshot_ms,
                prompt_render_ms=prompt_render_ms,
                llm_batch_ms=llm_batch_ms,
                action_decode_ms=action_decode_ms,
                environment_step_ms=environment_step_ms,
                total_ms=(perf_counter() - total_started_at) * 1000.0,
            )
        )
    return BatchedTextDecision(
        prompts=prompts,
        responses=responses,
        actions=tuple(decisions),
        metrics=tuple(metrics),
        fallback_used=cast(
            jax.Array,
            _place_for_environment(jnp.asarray(fallback, dtype=bool), device, enabled=place_inputs),
        ),
        transition=transition,
    )


def _collect_batched_text_rollout_impl(
    adapter: CraftaxAdapter[object, object, object],
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
    device_policy: EnvironmentDevicePolicy | None = None,
    profile: bool = False,
) -> ProfiledBatchedTextRollout:
    """Collect a fixed-horizon multi-env rollout and optionally profile every phase.

    Resets are evaluated for every row and selected only where terminal/truncated;
    this keeps the state PyTree static and avoids Python branching over JAX data.
    Each returned decision still records the pre-reset terminal transition.
    """
    if batch_size <= 0 or horizon <= 0:
        raise ValueError("batch_size and horizon must be positive")
    device = _resolve_device(device_policy)
    place_inputs = True if device_policy is None else device_policy.place_inputs
    keys = jax.random.split(jax.random.PRNGKey(seed), batch_size + horizon * 2 * batch_size)
    keys = cast(jax.Array, _place_for_environment(keys, device, enabled=place_inputs))
    batched_reset = _batched_reset(adapter, device_policy)
    reset = batched_reset(keys[:batch_size])
    state = reset.state
    action_mask = reset.action_mask
    dialogs: tuple[tuple[str, ...], ...] = tuple(() for _ in range(batch_size))
    decisions: list[BatchedTextDecision] = []
    reset_masks: list[jax.Array] = []
    timings: list[BatchedRolloutStepTiming] = []
    cursor = batch_size
    for _ in range(horizon):
        step_started_at = perf_counter() if profile else 0.0
        step_keys = keys[cursor : cursor + batch_size]
        reset_keys = keys[cursor + batch_size : cursor + 2 * batch_size]
        cursor += 2 * batch_size
        decision_timings: list[BatchedDecisionTiming] = []
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
            device_policy=device_policy,
            _inputs_are_placed=device is not None and place_inputs,
            _timing_sink=decision_timings.append if profile else None,
        )
        finished = jnp.logical_or(decision.transition.terminated, decision.transition.truncated)
        reset_started_at = perf_counter() if profile else 0.0
        fresh = batched_reset(reset_keys)
        if profile:
            jax.block_until_ready(fresh.state)
        reset_ms = (perf_counter() - reset_started_at) * 1000.0 if profile else 0.0
        replace_started_at = perf_counter() if profile else 0.0
        state = _replace_finished(decision.transition.state, fresh.state, finished)
        action_mask = cast(
            jax.Array,
            _replace_finished(decision.transition.action_mask, fresh.action_mask, finished),
        )
        if profile:
            action_mask.block_until_ready()
        replace_finished_ms = (
            (perf_counter() - replace_started_at) * 1000.0 if profile else 0.0
        )
        dialog_started_at = perf_counter() if profile else 0.0
        dialogs = tuple(
            () if bool(finished[index]) else (*dialogs[index], decision.responses[index].raw_text)
            for index in range(batch_size)
        )
        dialog_update_ms = (perf_counter() - dialog_started_at) * 1000.0 if profile else 0.0
        decisions.append(decision)
        reset_masks.append(finished)
        if profile:
            if len(decision_timings) != 1:
                raise RuntimeError("profiled decision did not report exactly one timing record")
            timings.append(
                BatchedRolloutStepTiming(
                    decision=decision_timings[0],
                    reset_ms=reset_ms,
                    replace_finished_ms=replace_finished_ms,
                    dialog_update_ms=dialog_update_ms,
                    total_ms=(perf_counter() - step_started_at) * 1000.0,
                )
            )
    return ProfiledBatchedTextRollout(
        BatchedTextRollout(reset, tuple(decisions), tuple(reset_masks)),
        tuple(timings),
    )


def collect_batched_text_rollout(
    adapter: CraftaxAdapter[object, object, object],
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
    device_policy: EnvironmentDevicePolicy | None = None,
) -> BatchedTextRollout:
    """Collect a fixed-horizon multi-env rollout with explicit done-reset semantics."""
    return _collect_batched_text_rollout_impl(
        adapter,
        renderer,
        backend,
        actions=actions,
        batch_size=batch_size,
        horizon=horizon,
        seed=seed,
        goal=goal,
        max_new_tokens=max_new_tokens,
        invalid_action=invalid_action,
        fallback_action_id=fallback_action_id,
        device_policy=device_policy,
        profile=False,
    ).rollout


def collect_batched_text_rollout_profiled(
    adapter: CraftaxAdapter[object, object, object],
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
    device_policy: EnvironmentDevicePolicy | None = None,
) -> ProfiledBatchedTextRollout:
    """Collect a rollout and return phase timings for sync bottleneck diagnosis."""
    return _collect_batched_text_rollout_impl(
        adapter,
        renderer,
        backend,
        actions=actions,
        batch_size=batch_size,
        horizon=horizon,
        seed=seed,
        goal=goal,
        max_new_tokens=max_new_tokens,
        invalid_action=invalid_action,
        fallback_action_id=fallback_action_id,
        device_policy=device_policy,
        profile=True,
    )


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
                    masked_action=metrics.masked_action,
                    fallback_used=bool(decision.fallback_used[environment_index]),
                    token_logprobs=response.token_logprobs,
                    token_ids=response.token_ids,
                    prompt_token_ids=response.prompt_token_ids,
                )
            )
        artifacts.append(ReplayArtifact(config_path, commit, backend, tuple(steps)))
    return tuple(artifacts)
