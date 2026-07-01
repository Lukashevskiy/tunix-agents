#!/usr/bin/env python3
"""Run one reproducible Qwen/Tunix/CrafText text episode and persist its evidence."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = "tunix-craftext.text-episode-metrics/v1"


def git_revision() -> str:
    """Return the current code revision without making a run depend on Git availability."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unversioned"


def episode_metrics(artifact: object) -> dict[str, object]:
    """Summarize replay outcomes without discarding per-step provenance.

    :param artifact: Replay artifact exposing its public steps and provenance fields.
    :returns: JSON-compatible aggregate metrics for one host-side text episode.
    """
    steps = getattr(artifact, "steps")
    token_logprobs = [
        logprob
        for step in steps
        for logprob in (getattr(step, "token_logprobs") or ())
    ]
    return {
        "schema": SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "config_path": getattr(artifact, "config_path"),
        "commit": getattr(artifact, "commit"),
        "backend": getattr(artifact, "backend"),
        "steps": len(steps),
        "reward_sum": sum(float(getattr(step, "reward")) for step in steps),
        "terminated": bool(steps[-1].terminated) if steps else False,
        "fallback_count": sum(bool(getattr(step, "fallback_used")) for step in steps),
        "invalid_format_count": sum(int(getattr(step, "invalid_format")) for step in steps),
        "unknown_action_count": sum(int(getattr(step, "unknown_action")) for step in steps),
        "generated_token_count": sum(len(getattr(step, "token_ids") or ()) for step in steps),
        "prompt_token_count": sum(
            len(getattr(step, "prompt_token_ids") or ()) for step in steps
        ),
        "mean_token_logprob": (
            sum(token_logprobs) / len(token_logprobs) if token_logprobs else None
        ),
    }


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Atomically persist one JSON artifact with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse one explicit episode run; no model download is implicit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/env/text/qwen_craftext.yaml"))
    parser.add_argument(
        "--snapshot", type=Path, default=Path("artifacts/models/qwen25-05b-instruct")
    )
    parser.add_argument("--cache-size", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument(
        "--replay-output",
        type=Path,
        default=Path("artifacts/trajectories/qwen-craftext-latest.json"),
    )
    parser.add_argument(
        "--metrics-output", type=Path, default=Path("artifacts/metrics/qwen-craftext-latest.json")
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> None:
    """Construct one real environment/model episode and write replay plus metrics evidence."""
    args = parse_args(arguments)
    from tunix_craftext.artifacts.replay import save_replay
    from tunix_craftext.env.config import load_mvp_config
    from tunix_craftext.env.prompts import MegaPromptRenderer
    from tunix_craftext.env.runtime import build_craftext_runtime
    from tunix_craftext.models.tunix_adapter import QwenTunixBackend
    from tunix_craftext.rollouts.text_episode import collect_text_episode

    config = load_mvp_config(args.config)
    if config.policy.implementation != "tunix":
        raise ValueError("text episode CLI requires policy.implementation: tunix")
    runtime = build_craftext_runtime(config)
    fallback_action_id = (
        runtime.actions.index_of("NOOP") if config.policy.invalid_action == "fallback" else None
    )
    artifact = collect_text_episode(
        runtime.adapter,
        MegaPromptRenderer(config.prompt.template),
        QwenTunixBackend(args.snapshot, cache_size=args.cache_size, seed=config.run.seed),
        goal="Stay alive and inspect the world.",
        actions=runtime.actions,
        horizon=args.horizon or config.environment.horizon,
        seed=config.run.seed,
        config_path=str(args.config),
        commit=git_revision(),
        max_new_tokens=args.max_new_tokens,
        invalid_action=config.policy.invalid_action,
        fallback_action_id=fallback_action_id,
    )
    save_replay(args.replay_output, artifact)
    write_json(args.metrics_output, episode_metrics(artifact))
    print(args.replay_output)
    print(args.metrics_output)


if __name__ == "__main__":
    main()
