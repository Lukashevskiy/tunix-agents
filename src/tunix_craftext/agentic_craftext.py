"""Serializable multi-turn CrafText environment for Tunix Agentic RL.

Tunix owns the conversation loop, trajectory tokenisation and training. This
module owns the CrafText episode state and exposes exactly one tool,
``craftext_step(action=...)``, for every environment transition.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax

from .adapters import CrafTextAdapter
from .config import load_mvp_config
from .prompts import ActionCatalog, MegaPromptRenderer, PromptContext, PromptRenderer
from .runtime import build_craftext_runtime

try:
    from tunix.rl.agentic.agents.tool_agent import ToolAgent  # type: ignore[import-untyped]
    from tunix.rl.agentic.environments.base_environment import (  # type: ignore[import-untyped]
        BaseTaskEnv,
        EnvStepResult,
    )
    from tunix.rl.agentic.tools.base_tool import BaseTool  # type: ignore[import-untyped]
except ImportError as error:  # pragma: no cover - exercised without the optional extra.
    raise RuntimeError(
        "install tunix-craftext[tunix] to use the multi-turn Tunix environment"
    ) from error

_LOGGER = logging.getLogger(__name__)
_TOOL_NAME = "craftext_step"


def _python_scalar(value: object) -> object:
    """Normalize NumPy/JAX scalar values emitted by Tunix dataset micro-batches."""
    item = getattr(value, "item", None)
    return item() if callable(item) else value


def agentic_task(*, goal: str, seed: int, horizon: int | None = None) -> dict[str, object]:
    """Create the serializable task payload consumed by ``CrafTextAgenticEnvironment``.

    ``group_id`` and ``pair_index`` are deliberately not accepted here: Tunix
    injects them when it creates multiple GRPO generations for the same task.
    """
    if not goal.strip():
        raise ValueError("goal must be non-empty")
    if isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    payload: dict[str, object] = {"goal": goal, "seed": int(seed)}
    if horizon is not None:
        if horizon <= 0:
            raise ValueError("horizon must be positive")
        payload["horizon"] = int(horizon)
    return payload


def agentic_environment_kwargs(config_path: Path | str) -> dict[str, str]:
    """Return JSON-serializable Tunix ``env_kwargs`` for a pinned MVP config."""
    return {"config_path": str(config_path)}


def _tool_payload(action: Any) -> tuple[str | None, str, str | None]:
    """Extract the call id and action label from one Tunix tool action."""
    if hasattr(action, "action"):
        action = action.action
    if not isinstance(action, list) or len(action) != 1 or not isinstance(action[0], Mapping):
        return None, "craftext-step", "expected exactly one craftext_step tool call"
    call = action[0]
    call_id = str(call.get("id", "craftext-step"))
    function = call.get("function")
    if not isinstance(function, Mapping) or function.get("name") != _TOOL_NAME:
        return None, call_id, f"expected {_TOOL_NAME} tool call"
    arguments = function.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None, call_id, "tool arguments must be JSON"
    if not isinstance(arguments, Mapping) or not isinstance(arguments.get("action"), str):
        return None, call_id, "tool arguments must include a string action"
    return arguments["action"], call_id, None


class CrafTextStepTool(BaseTool):
    """Submit one legal CrafText action and receive the next world state."""

    def get_json_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {"action": {"type": "string"}},
                    "required": ["action"],
                },
            },
        }

    def apply(self, **_: Any) -> Any:
        raise RuntimeError("craftext_step is executed by CrafTextAgenticEnvironment")


def build_craftext_tool_agent(system_prompt: str = "") -> ToolAgent:
    """Create Tunix's Qwen tool agent with the sole CrafText transition tool."""
    return ToolAgent(
        system_prompt=system_prompt,
        tool_parser_name="qwen",
        tool_map={_TOOL_NAME: CrafTextStepTool},
    )


class CrafTextAgenticEnvironment(BaseTaskEnv):
    """One serializable CrafText episode used by a Tunix Agentic rollout worker.

    Production callers pass only a task from :func:`agentic_task` and a
    ``config_path`` from :func:`agentic_environment_kwargs`. Optional component
    arguments exist solely for deterministic contract tests; they are not part
    of the distributed workload surface.
    """

    def __init__(
        self,
        task: Mapping[str, object],
        *,
        config_path: str | Path | None = None,
        adapter: CrafTextAdapter[object, object, object] | None = None,
        renderer: PromptRenderer[object] | None = None,
        actions: ActionCatalog | None = None,
        **kwargs: object,
    ) -> None:
        task_payload = dict(task)
        goal = _python_scalar(task_payload.get("goal"))
        seed = _python_scalar(task_payload.get("seed"))
        task_payload["goal"] = goal
        task_payload["seed"] = seed
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("task.goal must be a non-empty string")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("task.seed must be an integer")
        default_horizon, adapter, renderer, actions = self._resolve_components(
            config_path, adapter, renderer, actions
        )
        horizon = _python_scalar(task_payload.get("horizon", default_horizon))
        task_payload["horizon"] = horizon
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0:
            raise ValueError("task.horizon must be a positive integer")
        if len(actions.labels) != adapter.action_count:
            raise ValueError("action catalog length must equal adapter.action_count")
        super().__init__(task=task_payload, max_steps=horizon, **kwargs)
        self._adapter = adapter
        self._renderer = renderer
        self._actions = actions
        self._goal = goal
        self._seed = seed
        self._state: object | None = None
        self._action_mask: jax.Array | None = None

    @staticmethod
    def _resolve_components(
        config_path: str | Path | None,
        adapter: CrafTextAdapter[object, object, object] | None,
        renderer: PromptRenderer[object] | None,
        actions: ActionCatalog | None,
    ) -> tuple[int, CrafTextAdapter[object, object, object], PromptRenderer[object], ActionCatalog]:
        injected = (adapter, renderer, actions)
        if any(component is not None for component in injected):
            if config_path is not None or any(component is None for component in injected):
                raise ValueError(
                    "adapter, renderer and actions must be supplied together for test injection"
                )
            assert adapter is not None and renderer is not None and actions is not None
            return 1, adapter, renderer, actions
        if config_path is None:
            raise ValueError("config_path is required for a production agentic environment")
        config = load_mvp_config(Path(config_path))
        runtime = build_craftext_runtime(config)
        return (
            config.environment.horizon,
            runtime.adapter,
            MegaPromptRenderer(config.prompt.template),
            runtime.actions,
        )

    def _render(self) -> str:
        assert self._state is not None
        has_context = bool(getattr(self._adapter, "has_instruction_context", False))
        context = self._adapter.episode_context(self._state) if has_context else None
        prompt_state = (
            self._adapter.prompt_state(self._state)
            if hasattr(self._adapter, "prompt_state")
            else self._state
        )
        return self._renderer.render(
            PromptContext(
                context.instruction if context is not None else self._goal,
                context.env_state if context is not None else prompt_state,
                self._actions,
                safety="" if context is None else context.text_constraint,
                world_preset="" if context is None else context.world_preset,
            )
        ).text

    def _event(self, event: str, **fields: object) -> None:
        context = {
            "event": event,
            "group_id": self.extra_kwargs.get("group_id"),
            "pair_index": self.extra_kwargs.get("pair_index"),
            "step": self.step_count,
            **fields,
        }
        level = (
            logging.INFO
            if event in {"reset", "invalid_action"} or fields.get("done")
            else logging.DEBUG
        )
        _LOGGER.log(level, "agentic_craftext %s", context)

    def _initial_observation(self) -> dict[str, str]:
        reset = self._adapter.reset(jax.random.fold_in(jax.random.PRNGKey(self._seed), 0))
        self._state = reset.state
        self._action_mask = reset.action_mask
        self._event("reset", seed=self._seed, horizon=self.max_steps)
        return {"question": self._render()}

    def _invalid_result(self, call_id: str, reason: str) -> EnvStepResult:
        self._event("invalid_action", reason=reason)
        return EnvStepResult(
            observation={"tool_outputs": {call_id: f"Invalid action: {reason}"}},
            reward=0.0,
            done=False,
            info={"invalid_action": reason},
        )

    def _step_impl(self, action: Any) -> EnvStepResult:
        label, call_id, error = _tool_payload(action)
        if error is not None:
            return self._invalid_result(call_id, error)
        try:
            action_id = self._actions.index_of(label or "")
        except ValueError as error:
            return self._invalid_result(call_id, str(error))
        assert self._state is not None and self._action_mask is not None
        if not bool(self._action_mask[action_id]):
            return self._invalid_result(call_id, f"action {label!r} is unavailable in this state")
        transition = self._adapter.step(
            jax.random.fold_in(jax.random.PRNGKey(self._seed), self.step_count),
            self._state,
            action_id,
        )
        self._state = transition.state
        self._action_mask = transition.action_mask
        done = bool(transition.terminated) or bool(transition.truncated)
        self._event(
            "step",
            action_id=action_id,
            action_label=label,
            reward=float(transition.reward),
            done=done,
        )
        next_prompt = "Episode finished." if done else self._render()
        return EnvStepResult(
            observation={"tool_outputs": {call_id: next_prompt}},
            reward=float(transition.reward),
            done=done,
            info={"action_id": action_id, "action_label": label},
        )


def build_craftext_agentic_environment(
    adapter: CrafTextAdapter[object, object, object],
    renderer: PromptRenderer[object],
    *,
    goal: str,
    actions: ActionCatalog,
    horizon: int,
    seed: int,
) -> CrafTextAgenticEnvironment:
    """Build one deterministic test environment without a vendor/runtime dependency."""
    return CrafTextAgenticEnvironment(
        agentic_task(goal=goal, seed=seed, horizon=horizon),
        adapter=adapter,
        renderer=renderer,
        actions=actions,
    )
