"""Host-batched LLM to JAX-parallel environment transport tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tunix_craftext.adapters import CrafTextAdapter
from tunix_craftext.batched_rollout import (
    collect_batched_text_decision,
    collect_batched_text_rollout,
    replays_from_batched_rollout,
)
from tunix_craftext.llm import LlmResponse
from tunix_craftext.prompts import ActionCatalog, PromptContext, RenderedPrompt


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
