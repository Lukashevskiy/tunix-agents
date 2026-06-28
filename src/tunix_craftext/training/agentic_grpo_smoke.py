"""Scripted Agentic GRPO smoke rollout without model allocation.

This module verifies the part of GRPO that is independent from weights:
grouped tasks, multiple generations per task, tool-call environment turns,
episode rewards and group-normalized advantages.  It is not a replacement for
Tunix ``GRPOLearner``; it is the fast local gate before the real
``RLCluster(actor, rollout, reference)`` path allocates model memory.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..env.agentic_craftext import (
    CrafTextAgenticEnvironment,
    agentic_task,
    build_craftext_tool_agent,
)
from ..env.config import load_mvp_config

try:
    from tunix.rl.agentic.trajectory.trajectory_collect_engine import (  # type: ignore[import-untyped]
        TrajectoryCollectEngine,
    )
    from tunix.rl.rollout.base_rollout import RolloutOutput  # type: ignore[import-untyped]
except ImportError as error:  # pragma: no cover - optional Tunix extra.
    raise RuntimeError("install tunix-craftext[tunix] to use Agentic GRPO smoke") from error


@dataclass(frozen=True)
class ScriptedGenerationResult:
    """One scripted generation inside a GRPO task group."""

    group_id: int
    generation_id: int
    seed: int
    actions: tuple[str, ...]
    rewards: tuple[float, ...]
    total_reward: float
    advantage: float
    done: bool


def grouped_advantages(rewards: Sequence[float], *, epsilon: float = 1e-6) -> tuple[float, ...]:
    """Return GRPO-style group-normalized advantages for one task group."""
    if len(rewards) < 2:
        raise ValueError("GRPO grouped advantages require at least two generations")
    values = np.asarray(rewards, dtype=np.float32)
    std = float(np.std(values))
    if std < epsilon:
        return tuple(0.0 for _ in rewards)
    mean = float(np.mean(values))
    return tuple(float((value - mean) / (std + epsilon)) for value in values)


async def collect_scripted_grpo_group(
    *,
    config_path: Path,
    goal: str,
    seed: int,
    group_id: int,
    action_sequences: Sequence[Sequence[str]],
    horizon: int | None = None,
) -> tuple[ScriptedGenerationResult, ...]:
    """Collect one GRPO task group with deterministic scripted tool actions."""
    if len(action_sequences) < 2:
        raise ValueError("action_sequences must contain at least two generations")
    config = load_mvp_config(config_path)
    episode_horizon = config.environment.horizon if horizon is None else horizon
    results: list[ScriptedGenerationResult] = []
    totals: list[float] = []
    for generation_id, actions in enumerate(action_sequences):
        environment = CrafTextAgenticEnvironment(
            agentic_task(goal=goal, seed=seed, horizon=episode_horizon),
            config_path=str(config_path),
            group_id=group_id,
            pair_index=generation_id,
        )
        trajectory = await TrajectoryCollectEngine(
            build_craftext_tool_agent("Use craftext_step for every environment action."),
            environment,
            model_call=_scripted_tool_model_call(actions),
        ).collect(mode="Trajectory")
        rewards = tuple(float(step.reward) for step in trajectory.steps)
        total_reward = float(sum(rewards))
        totals.append(total_reward)
        results.append(
            ScriptedGenerationResult(
                group_id=group_id,
                generation_id=generation_id,
                seed=seed,
                actions=tuple(actions),
                rewards=rewards,
                total_reward=total_reward,
                advantage=0.0,
                done=bool(trajectory.steps[-1].done) if trajectory.steps else False,
            )
        )
    advantages = grouped_advantages(totals)
    return tuple(
        ScriptedGenerationResult(
            group_id=result.group_id,
            generation_id=result.generation_id,
            seed=result.seed,
            actions=result.actions,
            rewards=result.rewards,
            total_reward=result.total_reward,
            advantage=advantages[index],
            done=result.done,
        )
        for index, result in enumerate(results)
    )


def collect_scripted_grpo_group_sync(**kwargs: Any) -> tuple[ScriptedGenerationResult, ...]:
    """Synchronous wrapper for scripts and tests."""
    return asyncio.run(collect_scripted_grpo_group(**kwargs))


def save_scripted_grpo_smoke(path: Path, results: Sequence[ScriptedGenerationResult]) -> None:
    """Persist scripted GRPO smoke evidence as stable JSON."""
    payload = {
        "schema": "tunix-craftext.agentic-grpo-scripted-smoke/v1",
        "generations": [
            {
                "group_id": result.group_id,
                "generation_id": result.generation_id,
                "seed": result.seed,
                "actions": list(result.actions),
                "rewards": list(result.rewards),
                "total_reward": result.total_reward,
                "advantage": result.advantage,
                "done": result.done,
            }
            for result in results
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _scripted_tool_model_call(actions: Sequence[str]):
    iterator = iter(actions)

    def model_call(*_args: object, **_kwargs: object) -> RolloutOutput:
        try:
            label = next(iterator)
        except StopIteration:
            label = actions[-1]
        tool_call = {
            "name": "craftext_step",
            "arguments": {"action": label},
        }
        text = f"<tool_call>{json.dumps(tool_call, separators=(',', ':'))}</tool_call>"
        return RolloutOutput(
            text=[text],
            logits=[],
            tokens=[np.asarray([1], dtype=np.int32)],
            left_padded_prompt_tokens=np.asarray([[1]], dtype=np.int32),
            logprobs=[np.asarray([0.0], dtype=np.float32)],
        )

    return model_call
