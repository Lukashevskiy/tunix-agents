"""Host-batched LLM to JAX-parallel environment transport tests."""

from __future__ import annotations

import threading
import time

import jax
import jax.numpy as jnp
import pytest

import tunix_craftext.rollouts.batched as batched_module
from tunix_craftext.adapters import CrafTextAdapter
from tunix_craftext.env.prompts import ActionCatalog, PromptContext, RenderedPrompt
from tunix_craftext.models.llm import LlmResponse
from tunix_craftext.rollouts.batched import (
    EnvironmentDevicePolicy,
    HostBatchPolicy,
    collect_batched_text_decision,
    collect_batched_text_rollout,
    collect_batched_text_rollout_profiled,
    replays_from_batched_rollout,
)


class _Environment:
    def reset(self, key, params):
        del key, params
        return jnp.asarray(0), jnp.asarray(0)

    def step(self, key, state, action, params):
        del key, params
        next_state = state + action
        return next_state, next_state, jnp.asarray(action, dtype=jnp.float32), False, {}


class _Renderer:
    def render(self, context: PromptContext[object]) -> RenderedPrompt:
        return RenderedPrompt(f"state={context.observation}", context.actions, "test")


class _RecordingRenderer:
    def __init__(self) -> None:
        self.observations: list[object] = []

    def render(self, context: PromptContext[object]) -> RenderedPrompt:
        self.observations.append(context.observation)
        return RenderedPrompt(f"state={context.observation}", context.actions, "test")


class _SlowThreadRecordingRenderer:
    def __init__(self) -> None:
        self.thread_ids: list[int] = []

    def render(self, context: PromptContext[object]) -> RenderedPrompt:
        time.sleep(0.01)
        self.thread_ids.append(threading.get_ident())
        return RenderedPrompt(f"state={context.observation}", context.actions, "test")


class _Backend:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, request):
        return self.complete_batch((request,))[0]

    def complete_batch(self, requests):
        self.calls += 1
        return (
            LlmResponse("<action>DO</action>", "test", "test"),
            LlmResponse("invalid", "test", "test"),
        )


class _EchoBatchBackend:
    def complete(self, request):
        return self.complete_batch((request,))[0]

    def complete_batch(self, requests):
        return tuple(LlmResponse("<action>DO</action>", "test", "test") for _ in requests)


def test_batched_decision_uses_one_llm_batch_and_one_vmap_environment_step() -> None:
    """Valid and fallback decisions retain row order before parallel environment stepping."""
    adapter = CrafTextAdapter(_Environment(), None, action_count=2)
    backend = _Backend()
    catalog = ActionCatalog(("NOOP", "DO"))

    result = collect_batched_text_decision(
        adapter,
        _Renderer(),
        backend,
        states=jnp.asarray([10, 20]),
        action_masks=jnp.ones((2, 2), dtype=bool),
        actions=catalog,
        keys=jax.random.split(jax.random.PRNGKey(0), 2),
        goal="test",
        max_new_tokens=4,
        invalid_action="fallback",
        fallback_action_id=0,
    )

    assert backend.calls == 1
    assert [decision.action_id for decision in result.actions] == [1, 0]
    assert result.fallback_used.tolist() == [False, True]
    assert result.transition.state.tolist() == [11, 20]
    assert result.transition.reward.tolist() == [1.0, 0.0]


def test_batched_decision_can_render_prompts_with_ordered_host_threads() -> None:
    """Threaded host rendering overlaps prompt construction while preserving row order."""
    adapter = CrafTextAdapter(_Environment(), None, action_count=2)
    renderer = _SlowThreadRecordingRenderer()
    result = collect_batched_text_decision(
        adapter,
        renderer,
        _EchoBatchBackend(),
        states=jnp.asarray([10, 20, 30, 40]),
        action_masks=jnp.ones((4, 2), dtype=bool),
        actions=ActionCatalog(("NOOP", "DO")),
        keys=jax.random.split(jax.random.PRNGKey(0), 4),
        goal="test",
        max_new_tokens=4,
        invalid_action="fallback",
        fallback_action_id=0,
        host_batch_policy=HostBatchPolicy(prompt_workers=4),
    )

    assert [prompt.text for prompt in result.prompts] == [
        "state=10",
        "state=20",
        "state=30",
        "state=40",
    ]
    assert [decision.action_id for decision in result.actions] == [1, 1, 1, 1]
    assert len(set(renderer.thread_ids)) > 1


def test_host_batch_policy_rejects_non_positive_prompt_workers() -> None:
    """Thread policy fails early instead of silently falling back to serial render."""
    with pytest.raises(ValueError, match="prompt_workers must be positive"):
        HostBatchPolicy(prompt_workers=0)


def test_batched_decision_can_place_and_jit_environment_step_on_selected_device() -> None:
    """Accelerator lane keeps env keys/state/actions on an explicit JAX device."""
    adapter = CrafTextAdapter(_Environment(), None, action_count=2)
    device = jax.devices()[0]
    result = collect_batched_text_decision(
        adapter,
        _Renderer(),
        _Backend(),
        states=jnp.asarray([10, 20]),
        action_masks=jnp.ones((2, 2), dtype=bool),
        actions=ActionCatalog(("NOOP", "DO")),
        keys=jax.random.split(jax.random.PRNGKey(0), 2),
        goal="test",
        max_new_tokens=4,
        invalid_action="fallback",
        fallback_action_id=0,
        device_policy=EnvironmentDevicePolicy(
            backend=jax.default_backend(),
            device_index=0,
            jit_step=True,
        ),
    )

    assert result.transition.reward.tolist() == [1.0, 0.0]
    assert result.transition.reward.devices() == {device}
    assert result.transition.action_mask.devices() == {device}


def test_batched_decision_uses_one_host_snapshot_for_prompt_and_decode_inputs() -> None:
    """Prompt/decode sees host snapshots while env carry remains on the selected device."""
    adapter = CrafTextAdapter(_Environment(), None, action_count=2)
    renderer = _RecordingRenderer()
    result = collect_batched_text_decision(
        adapter,
        renderer,
        _Backend(),
        states=jax.device_put(jnp.asarray([10, 20]), jax.devices()[0]),
        action_masks=jax.device_put(jnp.ones((2, 2), dtype=bool), jax.devices()[0]),
        actions=ActionCatalog(("NOOP", "DO")),
        keys=jax.device_put(jax.random.split(jax.random.PRNGKey(0), 2), jax.devices()[0]),
        goal="test",
        max_new_tokens=4,
        invalid_action="fallback",
        fallback_action_id=0,
        device_policy=EnvironmentDevicePolicy(
            backend=jax.default_backend(),
            device_index=0,
            snapshot_prompt_inputs=True,
        ),
        _inputs_are_placed=True,
    )

    assert [int(value) for value in renderer.observations] == [10, 20]
    assert result.transition.reward.devices() == {jax.devices()[0]}


def test_batched_rollout_can_jit_reset_and_step_with_device_policy() -> None:
    """Full rollout path supports explicit colocated env reset/step execution."""
    adapter = CrafTextAdapter(_Environment(), None, action_count=2)
    rollout = collect_batched_text_rollout(
        adapter,
        _Renderer(),
        _Backend(),
        actions=ActionCatalog(("NOOP", "DO")),
        batch_size=2,
        horizon=2,
        seed=0,
        goal="test",
        max_new_tokens=4,
        invalid_action="fallback",
        fallback_action_id=0,
        device_policy=EnvironmentDevicePolicy(backend=jax.default_backend(), device_index=0),
    )

    assert len(rollout.decisions) == 2
    assert rollout.decisions[0].transition.reward.devices() == {jax.devices()[0]}


def test_batched_rollout_reuses_compiled_step_function(monkeypatch: pytest.MonkeyPatch) -> None:
    """Full rollout must not rebuild the vmapped/jitted env step inside every horizon step."""
    calls = 0
    original_batched_step = batched_module._batched_step

    def counting_batched_step(adapter, policy):
        nonlocal calls
        calls += 1
        return original_batched_step(adapter, policy)

    monkeypatch.setattr(batched_module, "_batched_step", counting_batched_step)

    adapter = CrafTextAdapter(_Environment(), None, action_count=2)
    rollout = collect_batched_text_rollout(
        adapter,
        _Renderer(),
        _Backend(),
        actions=ActionCatalog(("NOOP", "DO")),
        batch_size=2,
        horizon=3,
        seed=0,
        goal="test",
        max_new_tokens=4,
        invalid_action="fallback",
        fallback_action_id=0,
        device_policy=EnvironmentDevicePolicy(backend=jax.default_backend(), device_index=0),
    )

    assert len(rollout.decisions) == 3
    assert calls == 1


def test_batched_rollout_skips_reset_when_no_environment_finished(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-terminal steps should not pay a full batched reset every horizon step."""
    calls = 0
    original_batched_reset = batched_module._batched_reset

    def counting_batched_reset(adapter, policy):
        reset_fn = original_batched_reset(adapter, policy)

        def wrapped_reset(keys):
            nonlocal calls
            calls += 1
            return reset_fn(keys)

        return wrapped_reset

    monkeypatch.setattr(batched_module, "_batched_reset", counting_batched_reset)

    adapter = CrafTextAdapter(_Environment(), None, action_count=2)
    profiled = collect_batched_text_rollout_profiled(
        adapter,
        _Renderer(),
        _Backend(),
        actions=ActionCatalog(("NOOP", "DO")),
        batch_size=2,
        horizon=3,
        seed=0,
        goal="test",
        max_new_tokens=4,
        invalid_action="fallback",
        fallback_action_id=0,
    )

    assert len(profiled.rollout.decisions) == 3
    assert calls == 1
    assert profiled.phase_totals_ms()["reset_ms"] == 0.0
    assert profiled.phase_totals_ms()["replace_finished_ms"] == 0.0
    assert profiled.event_totals() == {"finished_count": 0, "reset_invocations": 0}


def test_profiled_batched_rollout_records_phase_timings() -> None:
    """Profiled rollout keeps the normal artifact and exposes per-phase timing totals."""
    adapter = CrafTextAdapter(_Environment(), None, action_count=2)
    profiled = collect_batched_text_rollout_profiled(
        adapter,
        _Renderer(),
        _Backend(),
        actions=ActionCatalog(("NOOP", "DO")),
        batch_size=2,
        horizon=2,
        seed=0,
        goal="test",
        max_new_tokens=4,
        invalid_action="fallback",
        fallback_action_id=0,
    )

    assert len(profiled.rollout.decisions) == 2
    assert len(profiled.timings) == 2
    totals = profiled.phase_totals_ms()
    events = profiled.event_totals()
    assert totals["prompt_snapshot_ms"] >= 0.0
    assert totals["prompt_render_ms"] >= 0.0
    assert totals["llm_batch_ms"] >= 0.0
    assert totals["action_decode_ms"] >= 0.0
    assert totals["environment_step_ms"] >= 0.0
    assert totals["reset_ms"] >= 0.0
    assert totals["replace_finished_ms"] >= 0.0
    assert totals["dialog_update_ms"] >= 0.0
    assert totals["total_ms"] >= totals["llm_batch_ms"]
    assert events["finished_count"] >= 0
    assert events["reset_invocations"] >= 0


def test_environment_device_policy_rejects_unavailable_backend() -> None:
    """Unavailable accelerator backends fail before a rollout silently falls back."""
    adapter = CrafTextAdapter(_Environment(), None, action_count=2)

    with pytest.raises(ValueError, match="backend is not available"):
        collect_batched_text_decision(
            adapter,
            _Renderer(),
            _Backend(),
            states=jnp.asarray([10, 20]),
            action_masks=jnp.ones((2, 2), dtype=bool),
            actions=ActionCatalog(("NOOP", "DO")),
            keys=jax.random.split(jax.random.PRNGKey(0), 2),
            goal="test",
            max_new_tokens=4,
            invalid_action="fallback",
            fallback_action_id=0,
            device_policy=EnvironmentDevicePolicy(backend="not-a-backend"),
        )


def test_batched_decision_falls_back_when_model_selects_masked_action() -> None:
    """Environment action masks are enforced before CrafText receives a step."""
    adapter = CrafTextAdapter(_Environment(), None, action_count=2)
    result = collect_batched_text_decision(
        adapter,
        _Renderer(),
        _Backend(),
        states=jnp.asarray([10, 20]),
        action_masks=jnp.asarray([[True, False], [True, True]], dtype=bool),
        actions=ActionCatalog(("NOOP", "DO")),
        keys=jax.random.split(jax.random.PRNGKey(0), 2),
        goal="test",
        max_new_tokens=4,
        invalid_action="fallback",
        fallback_action_id=0,
    )

    assert [decision.action_id for decision in result.actions] == [0, 0]
    assert result.metrics[0].masked_action == 1
    assert result.fallback_used.tolist() == [True, True]
    assert result.transition.reward.tolist() == [0.0, 0.0]


def test_batched_decision_rejects_masked_action_without_fallback() -> None:
    """Masked current-state actions fail loudly when no controlled fallback is configured."""
    adapter = CrafTextAdapter(_Environment(), None, action_count=2)

    with pytest.raises(ValueError, match="masked out"):
        collect_batched_text_decision(
            adapter,
            _Renderer(),
            _Backend(),
            states=jnp.asarray([10, 20]),
            action_masks=jnp.asarray([[True, False], [True, True]], dtype=bool),
            actions=ActionCatalog(("NOOP", "DO")),
            keys=jax.random.split(jax.random.PRNGKey(0), 2),
            goal="test",
            max_new_tokens=4,
        )


def test_batched_rollout_resets_only_finished_rows_and_exports_replays() -> None:
    """Terminal rows restart while other environments preserve their episode state."""
    adapter = CrafTextAdapter(_Environment(), None, action_count=2)
    catalog = ActionCatalog(("NOOP", "DO"))
    rollout = collect_batched_text_rollout(
        adapter,
        _Renderer(),
        _Backend(),
        actions=catalog,
        batch_size=2,
        horizon=2,
        seed=0,
        goal="test",
        max_new_tokens=4,
        invalid_action="fallback",
        fallback_action_id=0,
    )

    assert len(rollout.decisions) == 2
    assert len(rollout.reset_after_step) == 2
    replays = replays_from_batched_rollout(
        rollout, config_path="test", commit="abc", backend="test"
    )
    assert len(replays) == 2
    assert all(len(replay.steps) == 2 for replay in replays)


def test_batched_rollout_replay_feeds_token_ppo_training_path() -> None:
    """Replay export preserves token provenance and masks fallback rows before PPO loss."""
    from tunix_craftext.artifacts.text_trajectory import text_trajectory_from_replay
    from tunix_craftext.research.algorithms import masked_token_ppo_loss, masked_token_returns

    class TokenBackend(_Backend):
        def complete_batch(self, requests):
            self.calls += 1
            return (
                LlmResponse(
                    "<action>DO</action>",
                    "test",
                    "test",
                    token_ids=(11, 12),
                    token_logprobs=(-0.2, -0.3),
                    prompt_token_ids=(1, 2, 3),
                ),
                LlmResponse(
                    "invalid",
                    "test",
                    "test",
                    token_ids=(21,),
                    token_logprobs=(-0.7,),
                    prompt_token_ids=(4, 5),
                ),
            )

    adapter = CrafTextAdapter(_Environment(), None, action_count=2)
    rollout = collect_batched_text_rollout(
        adapter,
        _Renderer(),
        TokenBackend(),
        actions=ActionCatalog(("NOOP", "DO")),
        batch_size=2,
        horizon=1,
        seed=0,
        goal="test",
        max_new_tokens=4,
        invalid_action="fallback",
        fallback_action_id=0,
    )
    replays = replays_from_batched_rollout(
        rollout, config_path="test", commit="abc", backend="test"
    )
    token_batch = text_trajectory_from_replay(replays[1])

    assert token_batch.fallback_used.tolist() == [True]
    assert token_batch.policy_mask.tolist() == [[False]]
    token_batch = text_trajectory_from_replay(replays[0])
    returns = masked_token_returns(token_batch.rewards, token_batch.token_mask, gamma=0.99)
    zeros = jnp.zeros_like(token_batch.old_logprobs)
    loss, metrics = masked_token_ppo_loss(
        token_batch.old_logprobs,
        token_batch.old_logprobs,
        returns,
        zeros,
        zeros,
        returns,
        token_batch.policy_mask,
        clip_epsilon=0.2,
        value_coefficient=0.5,
        entropy=zeros,
        entropy_coefficient=0.01,
    )

    assert jnp.isfinite(loss)
    assert metrics["approx_kl"] == 0.0
