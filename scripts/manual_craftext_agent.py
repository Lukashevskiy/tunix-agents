#!/usr/bin/env python3
"""Manually control a CrafText agent and save every action as replay trajectory."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import jax
import jax.numpy as jnp

SCHEMA = "tunix-craftext.manual-craftext-metrics/v1"
DEFAULT_CONFIG = Path("configs/manual/caged_wood_achievements_energy.yaml")


@dataclass(frozen=True)
class ManualAction:
    """Parsed human action command."""

    action_id: int
    label: str


def git_revision() -> str:
    """Return current git revision or ``unversioned`` outside a git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unversioned"


def parse_manual_action(
    raw_value: str,
    *,
    labels: Sequence[str],
    action_mask: Sequence[bool],
) -> ManualAction | None:
    """Parse one human command into a legal CrafText action.

    ``q``, ``quit`` and ``exit`` return ``None``. Integer ids and case-insensitive
    action labels are accepted. Masked actions fail before the environment step.
    """
    value = raw_value.strip()
    if not value:
        raise ValueError("empty action; enter action id, label, or q")
    if value.lower() in {"q", "quit", "exit"}:
        return None
    if value.isdigit():
        action_id = int(value)
        if not 0 <= action_id < len(labels):
            raise ValueError(f"action id must be in [0, {len(labels) - 1}]")
    else:
        normalized = value.upper()
        lookup = {label.upper(): index for index, label in enumerate(labels)}
        if normalized not in lookup:
            raise ValueError(f"unknown action label: {value!r}")
        action_id = lookup[normalized]
    if not bool(action_mask[action_id]):
        raise ValueError(f"action {labels[action_id]} is currently masked out")
    return ManualAction(action_id=action_id, label=labels[action_id])


def legal_actions_text(labels: Sequence[str], action_mask: Sequence[bool]) -> str:
    """Return compact display text for currently legal actions."""
    return ", ".join(
        f"{index}:{label}"
        for index, (label, legal) in enumerate(zip(labels, action_mask, strict=True))
        if legal
    )


def manual_episode_metrics(artifact: object) -> dict[str, object]:
    """Summarize one manual replay while retaining the full replay separately."""
    steps = tuple(getattr(artifact, "steps"))
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "config_path": getattr(artifact, "config_path"),
        "commit": getattr(artifact, "commit"),
        "backend": getattr(artifact, "backend"),
        "steps": len(steps),
        "reward_sum": sum(float(step.reward) for step in steps),
        "terminated": bool(steps[-1].terminated) if steps else False,
        "truncated": bool(steps[-1].truncated) if steps else False,
        "manual_actions": [step.action_label for step in steps],
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write JSON atomically next to the target path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse manual control CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--goal", default="Follow the current CrafText scenario instruction.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument(
        "--replay-output",
        type=Path,
        default=Path("artifacts/trajectories/manual-craftext-latest.json"),
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=Path("artifacts/metrics/manual-craftext-latest.json"),
    )
    parser.add_argument(
        "--show-full-prompt",
        action="store_true",
        help="Print the full rendered MegaPrompt at every step.",
    )
    return parser.parse_args(arguments)


def collect_manual_episode(
    *,
    config_path: Path,
    goal: str,
    horizon: int | None,
    seed: int | None,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    show_full_prompt: bool = False,
) -> object:
    """Run the interactive manual control loop and return a replay artifact."""
    from tunix_craftext.config import load_mvp_config
    from tunix_craftext.prompts import (
        MegaPromptRenderer,
        PromptContext,
        compose_craftext_goal,
    )
    from tunix_craftext.replay import ReplayArtifact, ReplayStep
    from tunix_craftext.runtime import build_craftext_runtime

    config = load_mvp_config(config_path)
    runtime = build_craftext_runtime(config)
    renderer = MegaPromptRenderer(config.prompt.template)
    run_seed = config.run.seed if seed is None else seed
    limit = config.environment.horizon if horizon is None else horizon
    if limit <= 0:
        raise ValueError("horizon must be positive")

    keys = jax.random.split(jax.random.PRNGKey(run_seed), limit + 1)
    reset = runtime.adapter.reset(keys[0])
    state = reset.state
    action_mask = reset.action_mask
    steps: list[ReplayStep] = []

    for step_index in range(limit):
        context = (
            runtime.adapter.episode_context(state)
            if runtime.adapter.has_instruction_context
            else None
        )
        prompt_goal = (
            goal
            if context is None
            else compose_craftext_goal(
                goal,
                scenario_instruction=context.instruction,
                world_preset=context.world_preset,
                text_constraint=context.text_constraint,
            )
        )
        rendered = renderer.render(
            PromptContext(
                prompt_goal,
                runtime.adapter.prompt_state(state),
                runtime.actions,
                safety="" if context is None else context.text_constraint,
                world_preset="" if context is None else context.world_preset,
            )
        )

        legal = tuple(bool(value) for value in jnp.asarray(action_mask).tolist())
        print_fn("")
        print_fn(f"Step {step_index + 1}/{limit}")
        if context is not None:
            print_fn(f"Instruction: {context.instruction}")
            if context.text_constraint:
                print_fn(f"Constraint: {context.text_constraint}")
            print_fn(f"World preset: {context.world_preset}")
        print_fn(f"Legal actions: {legal_actions_text(runtime.actions.labels, legal)}")
        if show_full_prompt:
            print_fn("--- rendered prompt ---")
            print_fn(rendered.text)
            print_fn("--- end prompt ---")

        while True:
            try:
                parsed = parse_manual_action(
                    input_fn("action> "),
                    labels=runtime.actions.labels,
                    action_mask=legal,
                )
            except ValueError as error:
                print_fn(f"Invalid input: {error}")
                continue
            break
        if parsed is None:
            print_fn("Stopping manual episode.")
            break

        transition = runtime.adapter.step(keys[step_index + 1], state, parsed.action_id)
        state = transition.state
        action_mask = transition.action_mask
        reward = float(transition.reward)
        terminated = bool(transition.terminated)
        truncated = bool(transition.truncated)
        print_fn(
            f"Applied {parsed.action_id}:{parsed.label} -> reward={reward:.4f}, "
            f"terminated={terminated}, truncated={truncated}"
        )
        steps.append(
            ReplayStep(
                index=step_index,
                prompt=rendered.text,
                raw_completion=f"<manual_action>{parsed.label}</manual_action>",
                action_id=parsed.action_id,
                action_label=parsed.label,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                action_mask=legal,
                observation=transition.observation,
            )
        )
        if terminated or truncated:
            break

    return ReplayArtifact(
        config_path=str(config_path),
        commit=git_revision(),
        backend="manual-human",
        steps=tuple(steps),
        schema="tunix-craftext.manual-replay/v1",
    )


def main(arguments: Sequence[str] | None = None) -> None:
    """Run manual control and persist replay/metrics artifacts."""
    args = parse_args(arguments)
    from tunix_craftext.replay import save_replay

    artifact = collect_manual_episode(
        config_path=args.config,
        goal=args.goal,
        horizon=args.horizon,
        seed=args.seed,
        show_full_prompt=args.show_full_prompt,
    )
    save_replay(args.replay_output, artifact)
    write_json(args.metrics_output, manual_episode_metrics(artifact))
    print(f"replay: {args.replay_output}")
    print(f"metrics: {args.metrics_output}")
    print(
        "visualize: PYTHONPATH=src .venv/bin/python "
        f"scripts/visualize_trajectory.py --trajectory {args.replay_output}"
    )


if __name__ == "__main__":
    main()
