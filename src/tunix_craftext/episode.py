"""Host-side prompt/LLM/environment orchestration with inspectable replay output."""

from __future__ import annotations

import jax

from .adapters import CrafTextAdapter
from .llm import LlmBackend, LlmRequest
from .prompts import ActionCatalog, PromptContext, PromptRenderer
from .replay import ReplayArtifact, ReplayStep
from .text_policy import decode_action


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
    :returns: Ordered replay, including every rendered prompt and raw completion.
    :raises ValueError: If horizon is non-positive or catalog disagrees with the adapter.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if len(actions.labels) != adapter.action_count:
        raise ValueError("action catalog length must equal adapter action_count")

    keys = jax.random.split(jax.random.PRNGKey(seed), horizon + 1)
    reset = adapter.reset(keys[0])
    state, observation = reset.state, reset.observation
    dialog: tuple[str, ...] = ()
    steps: list[ReplayStep] = []
    backend_name = "unknown"

    for index, key in enumerate(keys[1:]):
        prompt = renderer.render(PromptContext(goal, observation, actions, dialog))
        response = backend.complete(LlmRequest(prompt))
        decision, _ = decode_action(prompt, response.raw_text)
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
            )
        )
        backend_name = f"{response.backend}:{response.model}"
        dialog = (*dialog, response.raw_text)
        state, observation = transition.state, transition.observation
        if terminated or bool(transition.truncated):
            break

    return ReplayArtifact(
        config_path=config_path, commit=commit, backend=backend_name, steps=tuple(steps)
    )
