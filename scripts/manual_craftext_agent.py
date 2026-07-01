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
DEFAULT_CONFIG = Path("configs/env/manual/caged_wood_achievements_energy.yaml")
BLOCK_SYMBOLS = {
    0: "?",
    1: "#",
    2: ".",
    3: "~",
    4: "S",
    5: "T",
    6: "W",
    7: ",",
    8: "C",
    9: "I",
    10: "D",
    11: "B",
    12: "F",
    13: ":",
    14: "L",
    15: "p",
    16: "P",
    17: "#",
    18: " ",
    19: "m",
    20: "^",
    21: "s",
    22: "r",
    23: "$",
    24: "f",
}
FACING_NAMES = {
    0: "NOOP",
    1: "LEFT",
    2: "RIGHT",
    3: "UP",
    4: "DOWN",
}
INVENTORY_FIELDS = (
    "wood",
    "stone",
    "coal",
    "iron",
    "diamond",
    "sapling",
    "wood_pickaxe",
    "stone_pickaxe",
    "iron_pickaxe",
    "wood_sword",
    "stone_sword",
    "iron_sword",
)


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


def observation_text(observation: object, *, max_values: int = 24) -> str:
    """Return a compact terminal-safe summary of the current environment observation.

    Pixel observations can be large; printing the full nested array makes manual
    control unusable.  The summary keeps enough information for sanity checking
    while the complete observation is still saved in replay JSON.
    """
    try:
        array = jnp.asarray(observation)
    except Exception:
        return f"Observation: {type(observation).__name__}"
    if array.size == 0:
        return f"Observation: shape={tuple(array.shape)}, dtype={array.dtype}, empty"

    flat = jnp.ravel(array)
    preview_values = [format_observation_value(value) for value in flat[:max_values].tolist()]
    suffix = " ..." if int(flat.size) > max_values else ""
    return (
        f"Observation: shape={tuple(array.shape)}, dtype={array.dtype}, "
        f"min={format_observation_value(jnp.min(flat).item())}, "
        f"max={format_observation_value(jnp.max(flat).item())}, "
        f"preview=[{', '.join(preview_values)}{suffix}]"
    )


def format_observation_value(value: object) -> str:
    """Format one scalar observation value for compact CLI output."""
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)


def manual_state_text(env_state: object, *, radius: int = 8) -> str:
    """Render a concrete tactical map and scalar state for manual decisions."""
    if not hasattr(env_state, "map") or not hasattr(env_state, "player_position"):
        return "Map view: unavailable for this environment state"
    return "\n".join(
        (
            vital_text(env_state),
            inventory_text(env_state),
            (
                "Map legend: @ player, T tree, . grass, # wall/oob, ~ water, "
                "S stone, C coal, I iron, D diamond, p plant"
            ),
            ascii_map_text(env_state, radius=radius),
        )
    )


def vital_text(env_state: object) -> str:
    """Format player scalar state used for Caged constraints."""
    direction = scalar_int(getattr(env_state, "player_direction", None))
    fields = [
        ("pos", tuple(int(value) for value in jnp.asarray(env_state.player_position).tolist())),
        ("facing", FACING_NAMES.get(direction, str(direction))),
        ("hp", scalar_int(getattr(env_state, "player_health", None))),
        ("food", scalar_int(getattr(env_state, "player_food", None))),
        ("drink", scalar_int(getattr(env_state, "player_drink", None))),
        ("energy", scalar_int(getattr(env_state, "player_energy", None))),
        ("sleeping", scalar_bool(getattr(env_state, "is_sleeping", None))),
        ("timestep", scalar_int(getattr(env_state, "timestep", None))),
    ]
    return "Vitals: " + ", ".join(f"{name}={value}" for name, value in fields if value is not None)


def inventory_text(env_state: object) -> str:
    """Format non-empty inventory items for terminal display."""
    inventory = getattr(env_state, "inventory", None)
    if inventory is None:
        return "Inventory: unavailable"
    non_empty = []
    for field in INVENTORY_FIELDS:
        if not hasattr(inventory, field):
            continue
        value = scalar_int(getattr(inventory, field))
        if value:
            non_empty.append(f"{field}={value}")
    return "Inventory: " + (", ".join(non_empty) if non_empty else "empty")


def ascii_map_text(env_state: object, *, radius: int = 8) -> str:
    """Render a local ASCII map around the player from Craftax map tensors."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    map_array = jnp.asarray(env_state.map)
    if map_array.ndim != 2:
        return f"Map view: unsupported map shape={tuple(map_array.shape)}"
    player_row, player_col = (
        int(value) for value in jnp.asarray(env_state.player_position).tolist()
    )
    height, width = int(map_array.shape[0]), int(map_array.shape[1])
    overlays = mob_overlays(env_state)
    rows: list[str] = []
    for row in range(player_row - radius, player_row + radius + 1):
        chars: list[str] = []
        for col in range(player_col - radius, player_col + radius + 1):
            if row == player_row and col == player_col:
                chars.append("@")
            elif (row, col) in overlays:
                chars.append(overlays[(row, col)])
            elif row < 0 or col < 0 or row >= height or col >= width:
                chars.append("#")
            else:
                chars.append(BLOCK_SYMBOLS.get(int(map_array[row, col]), "?"))
        rows.append("".join(chars))
    return "Map view:\n" + "\n".join(rows)


def mob_overlays(env_state: object) -> dict[tuple[int, int], str]:
    """Return active mob positions to overlay on top of the terrain map."""
    overlays: dict[tuple[int, int], str] = {}
    for attr, marker in (("zombies", "Z"), ("skeletons", "K"), ("cows", "c")):
        mob_state = getattr(env_state, attr, None)
        if (
            mob_state is None
            or not hasattr(mob_state, "position")
            or not hasattr(mob_state, "mask")
        ):
            continue
        positions = jnp.asarray(mob_state.position).tolist()
        masks = jnp.asarray(mob_state.mask).tolist()
        for position, active in zip(positions, masks, strict=False):
            if active:
                overlays[(int(position[0]), int(position[1]))] = marker
    return overlays


def scalar_int(value: object) -> int | None:
    """Convert optional scalar JAX/Python values to int."""
    if value is None:
        return None
    return int(jnp.asarray(value).item())


def scalar_bool(value: object) -> bool | None:
    """Convert optional scalar JAX/Python values to bool."""
    if value is None:
        return None
    return bool(jnp.asarray(value).item())


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
    from tunix_craftext.artifacts.replay import ReplayArtifact, ReplayStep
    from tunix_craftext.env.config import load_mvp_config
    from tunix_craftext.env.prompts import (
        MegaPromptRenderer,
        PromptContext,
        compose_craftext_goal,
    )
    from tunix_craftext.env.runtime import build_craftext_runtime

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
    observation = reset.observation
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
        state_view = (
            manual_state_text(context.env_state)
            if context is not None
            else observation_text(observation)
        )
        print_fn(state_view)
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
        observation = transition.observation
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
    from tunix_craftext.artifacts.replay import save_replay

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
        "visualize: uv run python "
        f"scripts/visualize_trajectory.py --trajectory {args.replay_output}"
    )


if __name__ == "__main__":
    main()
