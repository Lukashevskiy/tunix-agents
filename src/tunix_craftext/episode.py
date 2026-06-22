"""Host-side prompt/LLM/environment orchestration with inspectable replay output."""

from __future__ import annotations

from typing import Literal

import jax

from .adapters import CrafTextAdapter
from .llm import LlmBackend, LlmRequest
from .prompts import ActionCatalog, PromptContext, PromptRenderer
from .replay import ReplayArtifact, ReplayStep
from .text_policy import DecodedAction, decode_action, decode_action_outcome


def collect_text_episode(
    adapter: CrafTextAdapter[object, object, object],
    renderer: PromptRenderer[object],
    backend: LlmBackend,
    *,
    goal: str,
    actions: ActionCatalog,
    horizon: int,
    seed: int,
    config_path: str,
    commit: str,
    max_new_tokens: int = 32,
    invalid_action: Literal["error", "fallback"] = "error",
    fallback_action_id: int | None = None,
) -> ReplayArtifact:
    """Run prompt → completion → strict action decode → environment steps.

    :param adapter: Environment boundary to reset and step.
    :param renderer: Typed prompt renderer.
    :param backend: Model backend returning raw completions.
    :param goal: Episode instruction included in each rendered prompt.
    :param actions: Model-visible labels aligned with environment action ids.
    :param horizon: Maximum number of host-side decisions.
    :param seed: Deterministic source of reset and step keys.
    :param config_path: Config provenance written into replay.
    :param commit: Code revision provenance written into replay.
    :param max_new_tokens: Per-decision completion cap passed to the model backend.
    :param invalid_action: Whether an invalid model completion aborts or takes a declared fallback.
    :param fallback_action_id: Required action id when ``invalid_action`` is ``"fallback"``.
    :returns: Ordered replay, including every rendered prompt and raw completion.
    :raises ValueError: If horizon is non-positive or catalog disagrees with the adapter.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if len(actions.labels) != adapter.action_count:
        raise ValueError("action catalog length must equal adapter action_count")
    if invalid_action not in {"error", "fallback"}:
        raise ValueError("invalid_action must be 'error' or 'fallback'")
    if invalid_action == "fallback" and (
        fallback_action_id is None or not 0 <= fallback_action_id < len(actions.labels)
    ):
        raise ValueError("fallback action id must be inside the action catalog")

    keys = jax.random.split(jax.random.PRNGKey(seed), horizon + 1)
    reset = adapter.reset(keys[0])
    state = reset.state
    dialog: tuple[str, ...] = ()
    steps: list[ReplayStep] = []
    backend_name = "unknown"

    for index, key in enumerate(keys[1:]):
        # MegaPrompts renders CrafText's structured ``EnvState`` (map, inventory,
        # coordinates), while ``observation`` remains the numerical policy input.
        prompt = renderer.render(PromptContext(goal, state, actions, dialog))
        response = backend.complete(LlmRequest(prompt, max_new_tokens=max_new_tokens))
        decision, metrics = decode_action_outcome(prompt, response.raw_text)
        fallback_used = False
        if decision is None:
            if invalid_action == "error":
                decode_action(prompt, response.raw_text)
                raise AssertionError("decode_action must raise for an invalid outcome")
            assert fallback_action_id is not None
            decision = DecodedAction(
                action_id=fallback_action_id,
                label=actions.labels[fallback_action_id],
                raw_text=response.raw_text,
            )
            fallback_used = True
        transition = adapter.step(key, state, decision.action_id)
        terminated = bool(transition.terminated)
        steps.append(
            ReplayStep(
                index=index,
                prompt=prompt.text,
                raw_completion=response.raw_text,
                action_id=decision.action_id,
                action_label=decision.label,
                reward=float(transition.reward),
                terminated=terminated,
                invalid_format=metrics.invalid_format,
                unknown_action=metrics.unknown_action,
                fallback_used=fallback_used,
                token_logprobs=response.token_logprobs,
                token_ids=response.token_ids,
            )
        )
        backend_name = f"{response.backend}:{response.model}"
        dialog = (*dialog, response.raw_text)
        state = transition.state
        if terminated or bool(transition.truncated):
            break

    return ReplayArtifact(
        config_path=config_path, commit=commit, backend=backend_name, steps=tuple(steps)
    )
