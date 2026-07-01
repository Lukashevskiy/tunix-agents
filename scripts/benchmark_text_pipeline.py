#!/usr/bin/env python3
"""Measure and trace the real MegaPrompts → Qwen → CrafText decision pipeline.

The tool keeps model downloads implicit, separates warmup from recorded repeats,
and persists each decision's phase timings alongside aggregate median/p95 values.
It measures a host-side synchronous LLM path; do not compare its result with a
compiled environment-only throughput benchmark.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

SCHEMA = "tunix-craftext.text-pipeline-benchmark/v1"
PHASES = (
    "prompt_render_ms",
    "llm_generation_ms",
    "action_decode_ms",
    "environment_step_ms",
    "decision_total_ms",
)


def percentile(samples: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for non-empty timing samples."""
    if not samples:
        raise ValueError("percentile requires at least one sample")
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(samples)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_samples(samples: Sequence[float]) -> dict[str, float | int]:
    """Summarize one recorded phase in milliseconds with stable rounding.

    :param samples: Blocking wall-clock timing samples in milliseconds.
    :returns: Count, mean, median, p95, minimum and maximum.
    """
    if not samples:
        raise ValueError("phase requires at least one recorded timing")
    return {
        "count": len(samples),
        "mean_ms": round(sum(samples) / len(samples), 6),
        "median_ms": round(percentile(samples, 0.5), 6),
        "p95_ms": round(percentile(samples, 0.95), 6),
        "min_ms": round(min(samples), 6),
        "max_ms": round(max(samples), 6),
    }


def git_revision() -> str:
    """Return the code revision without requiring Git for a benchmark run."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unversioned"


def git_dirty() -> bool:
    """Return whether the repository has uncommitted work when Git is available."""
    try:
        return bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return False


def now_ms() -> float:
    """Return a monotonic wall-clock marker in milliseconds."""
    return time.perf_counter() * 1_000.0


def run_episode(
    *,
    adapter: object,
    renderer: object,
    backend: object,
    actions: object,
    horizon: int,
    seed: int,
    max_new_tokens: int,
    run_index: int,
    record_trace: bool,
) -> list[dict[str, object]]:
    """Time every host decision in one real environment episode.

    Imports stay inside the function so ``--help`` and unit tests do not initialize
    JAX, CrafText, or model dependencies.
    """
    import jax

    from tunix_craftext.env.prompts import PromptContext
    from tunix_craftext.env.text_policy import DecodedAction, decode_action_outcome
    from tunix_craftext.models.llm import LlmRequest

    action_labels = getattr(actions, "labels")
    fallback_action_id = actions.index_of("NOOP")
    keys = jax.random.split(jax.random.PRNGKey(seed), horizon + 1)
    reset = adapter.reset(keys[0])
    state = reset.state
    dialog: tuple[str, ...] = ()
    trace: list[dict[str, object]] = []
    for step_index, key in enumerate(keys[1:]):
        total_started = now_ms()
        started = now_ms()
        prompt = renderer.render(
            PromptContext(
                "Stay alive, inspect the world, choose one valid action.", state, actions, dialog
            )
        )
        prompt_render_ms = now_ms() - started

        started = now_ms()
        response = backend.complete(LlmRequest(prompt, max_new_tokens=max_new_tokens))
        llm_generation_ms = now_ms() - started

        started = now_ms()
        decision, metrics = decode_action_outcome(prompt, response.raw_text)
        fallback_used = decision is None
        if decision is None:
            decision = DecodedAction(
                action_id=fallback_action_id,
                label=action_labels[fallback_action_id],
                raw_text=response.raw_text,
            )
        action_decode_ms = now_ms() - started

        started = now_ms()
        transition = adapter.step(key, state, decision.action_id)
        environment_step_ms = now_ms() - started
        decision_total_ms = now_ms() - total_started

        if record_trace:
            trace.append(
                {
                    "run": run_index,
                    "step": step_index,
                    "prompt_render_ms": round(prompt_render_ms, 6),
                    "llm_generation_ms": round(llm_generation_ms, 6),
                    "action_decode_ms": round(action_decode_ms, 6),
                    "environment_step_ms": round(environment_step_ms, 6),
                    "decision_total_ms": round(decision_total_ms, 6),
                    "prompt_characters": len(prompt.text),
                    "prompt_token_count": len(response.prompt_token_ids or ()),
                    "generated_token_count": len(response.token_ids or ()),
                    "action_id": decision.action_id,
                    "action_label": decision.label,
                    "reward": float(transition.reward),
                    "terminated": bool(transition.terminated),
                    "truncated": bool(transition.truncated),
                    "fallback_used": fallback_used,
                    "invalid_format": metrics.invalid_format,
                    "unknown_action": metrics.unknown_action,
                }
            )
        dialog = (*dialog, response.raw_text)
        state = transition.state
        if bool(transition.terminated) or bool(transition.truncated):
            break
    return trace


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Atomically persist one benchmark evidence artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit model, trace and output settings for one benchmark run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/env/text/qwen_craftext.yaml"))
    parser.add_argument(
        "--snapshot", type=Path, default=Path("artifacts/models/qwen25-05b-instruct")
    )
    parser.add_argument("--cache-size", type=int, default=2048)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--isolate-runs",
        action="store_true",
        help=(
            "Run each warmup/repeat in a child process and retain partial evidence "
            "on native failure."
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/benchmarks/text-pipeline-latest.json")
    )
    parser.add_argument("--_single-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--_single-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--_run-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--_record-trace", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(arguments)


def run_isolated_repeats(args: argparse.Namespace) -> None:
    """Run each episode in a fresh process and retain partial trace evidence.

    This protects long local Qwen measurements from native JAX exits that cannot
    be caught by Python. A failed child is recorded rather than silently losing
    earlier samples.
    """
    trace: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    total_runs = args.warmup_runs + args.repeats
    for ordinal in range(total_runs):
        recording = ordinal >= args.warmup_runs
        output = args.output.with_suffix(f".run-{ordinal:03d}.json")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--_single-run",
            "--config",
            str(args.config),
            "--snapshot",
            str(args.snapshot),
            "--cache-size",
            str(args.cache_size),
            "--horizon",
            str(args.horizon),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--_run-index",
            str(ordinal),
            "--_single-output",
            str(output),
        ]
        if recording:
            command.append("--_record-trace")
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        try:
            if completed.returncode == 0 and output.is_file():
                payload = json.loads(output.read_text(encoding="utf-8"))
                child_trace = payload.get("trace") if isinstance(payload, dict) else None
                if isinstance(child_trace, list) and recording:
                    trace.extend(item for item in child_trace if isinstance(item, dict))
                continue
            failures.append(
                {
                    "run": ordinal,
                    "recording": recording,
                    "exit_code": completed.returncode,
                    "detail": (completed.stderr.strip() or completed.stdout.strip())[-1_000:],
                }
            )
        finally:
            output.unlink(missing_ok=True)
        write_json(
            args.output,
            {
                "schema": SCHEMA,
                "timestamp": datetime.now(UTC).isoformat(),
                "git_revision": git_revision(),
                "git_dirty": git_dirty(),
                "status": "partial" if trace else "failed",
                "measurement": {"isolate_runs": True, "failures": failures},
                "trace": trace,
            },
        )
    if not trace:
        raise RuntimeError("no recorded benchmark decisions completed; inspect artifact failures")
    phase_summaries = {
        phase: summarize_samples([float(record[phase]) for record in trace]) for phase in PHASES
    }
    write_json(
        args.output,
        {
            "schema": SCHEMA,
            "timestamp": datetime.now(UTC).isoformat(),
            "git_revision": git_revision(),
            "git_dirty": git_dirty(),
            "status": "partial" if failures else "ok",
            "config": str(args.config),
            "config_sha256": sha256(args.config.read_bytes()).hexdigest(),
            "model_snapshot": str(args.snapshot),
            "parameters": {
                "horizon": args.horizon,
                "warmup_runs": args.warmup_runs,
                "repeats": args.repeats,
                "max_new_tokens": args.max_new_tokens,
                "cache_size": args.cache_size,
            },
            "hardware": {"platform": platform.platform(), "jax_backend": "recorded by child"},
            "measurement": {
                "scope": "host-side synchronous Qwen/Tunix completion and CrafText step",
                "isolate_runs": True,
                "failures": failures,
            },
            "phase_summaries": phase_summaries,
            "trace": trace,
        },
    )


def main(arguments: Sequence[str] | None = None) -> None:
    """Run warmup and recorded real LLM/environment pipeline decisions."""
    args = parse_args(arguments)
    if args.horizon <= 0 or args.warmup_runs < 0 or args.repeats <= 0 or args.max_new_tokens <= 0:
        raise ValueError(
            "horizon/repeats/max-new-tokens must be positive; warmup-runs non-negative"
        )
    if not args.snapshot.is_dir():
        raise FileNotFoundError(f"Expected explicit local model snapshot at {args.snapshot}")
    if args.isolate_runs and not args._single_run:
        run_isolated_repeats(args)
        print(args.output)
        return

    import jax

    from tunix_craftext.env.config import load_mvp_config
    from tunix_craftext.env.prompts import MegaPromptRenderer
    from tunix_craftext.env.runtime import build_craftext_runtime
    from tunix_craftext.models.tunix_adapter import QwenTunixBackend

    config = load_mvp_config(args.config)
    if config.policy.implementation != "tunix":
        raise ValueError("text pipeline benchmark requires policy.implementation: tunix")
    runtime = build_craftext_runtime(config)
    renderer = MegaPromptRenderer(config.prompt.template)
    backend = QwenTunixBackend(args.snapshot, cache_size=args.cache_size, seed=config.run.seed)

    if args._single_run:
        if args._single_output is None:
            raise ValueError("--_single-run requires --_single-output")
        trace = run_episode(
            adapter=runtime.adapter,
            renderer=renderer,
            backend=backend,
            actions=runtime.actions,
            horizon=args.horizon,
            seed=config.run.seed + args._run_index,
            max_new_tokens=args.max_new_tokens,
            run_index=args._run_index,
            record_trace=args._record_trace,
        )
        write_json(args._single_output, {"trace": trace})
        return

    for warmup in range(args.warmup_runs):
        run_episode(
            adapter=runtime.adapter,
            renderer=renderer,
            backend=backend,
            actions=runtime.actions,
            horizon=args.horizon,
            seed=config.run.seed + warmup,
            max_new_tokens=args.max_new_tokens,
            run_index=warmup,
            record_trace=False,
        )

    trace: list[dict[str, object]] = []
    for repeat in range(args.repeats):
        trace.extend(
            run_episode(
                adapter=runtime.adapter,
                renderer=renderer,
                backend=backend,
                actions=runtime.actions,
                horizon=args.horizon,
                seed=config.run.seed + args.warmup_runs + repeat,
                max_new_tokens=args.max_new_tokens,
                run_index=repeat,
                record_trace=True,
            )
        )

    phase_summaries = {
        phase: summarize_samples([float(record[phase]) for record in trace]) for phase in PHASES
    }
    payload: dict[str, object] = {
        "schema": SCHEMA,
        "status": "ok",
        "timestamp": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(),
        "git_dirty": git_dirty(),
        "config": str(args.config),
        "config_sha256": sha256(args.config.read_bytes()).hexdigest(),
        "model_snapshot": str(args.snapshot),
        "parameters": {
            "horizon": args.horizon,
            "warmup_runs": args.warmup_runs,
            "repeats": args.repeats,
            "max_new_tokens": args.max_new_tokens,
            "cache_size": args.cache_size,
        },
        "hardware": {
            "platform": platform.platform(),
            "jax_backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "jax_version": version("jax"),
        },
        "measurement": {
            "scope": "host-side synchronous Qwen/Tunix completion and CrafText step",
            "excluded": "model download, initial backend construction, warmup decisions",
            "timing_clock": "time.perf_counter",
        },
        "phase_summaries": phase_summaries,
        "trace": trace,
    }
    write_json(args.output, payload)
    print(args.output)


if __name__ == "__main__":
    main()
