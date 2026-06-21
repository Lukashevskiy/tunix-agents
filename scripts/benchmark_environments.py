#!/usr/bin/env python3
"""Measure compiled CrafText rollouts and persist reproducible matrix evidence.

Compilation and steady-state execution are intentionally separate.  Every matrix
point executes one blocking warmup/compile invocation and then a configurable
number of blocking steady-state invocations.  The JSON artifact retains raw
samples so its median and p95 can be independently audited later.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

SCHEMA = "tunix-craftext.environment-benchmark/v2"
BenchmarkPoint = dict[str, object]


def percentile(samples: Sequence[float], quantile: float) -> float:
    """Return a linearly interpolated percentile for one non-empty sample sequence.

    :param samples: Measured wall-clock samples in milliseconds.
    :param quantile: Percentile in the inclusive range ``[0, 1]``.
    :returns: Interpolated percentile in milliseconds.
    :raises ValueError: If samples are empty or quantile is outside its valid range.
    """
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


def summarize_samples(samples_ms: Sequence[float]) -> dict[str, float]:
    """Return stable timing statistics for a sequence of steady-state measurements.

    :param samples_ms: Blocking measured executions in milliseconds.
    :returns: Mean, median, p95, minimum and maximum milliseconds.
    """
    if not samples_ms:
        raise ValueError("benchmark requires at least one steady-state repeat")
    return {
        "steady_state_mean_ms": round(sum(samples_ms) / len(samples_ms), 6),
        "steady_state_median_ms": round(percentile(samples_ms, 0.5), 6),
        "steady_state_p95_ms": round(percentile(samples_ms, 0.95), 6),
        "steady_state_min_ms": round(min(samples_ms), 6),
        "steady_state_max_ms": round(max(samples_ms), 6),
    }


def add_baseline_comparisons(
    points: Sequence[BenchmarkPoint], baseline_variant: str = "craftext-full"
) -> None:
    """Annotate points with median-throughput ratio against the matching baseline.

    Comparison is only valid within an identical ``(batch_size, horizon)`` pair.

    :param points: Mutable benchmark records, including successful and failed points.
    :param baseline_variant: Variant used as the comparison reference.
    """
    baselines: dict[tuple[object, object], float] = {}
    for point in points:
        if point.get("variant") != baseline_variant or point.get("status") == "failed":
            continue
        throughput = point.get("env_steps_per_second_median")
        if isinstance(throughput, (int, float)) and throughput > 0:
            baselines[(point.get("batch_size"), point.get("horizon"))] = float(throughput)

    for point in points:
        if point.get("status") == "failed":
            continue
        throughput = point.get("env_steps_per_second_median")
        baseline = baselines.get((point.get("batch_size"), point.get("horizon")))
        if not isinstance(throughput, (int, float)) or baseline is None:
            continue
        relative = float(throughput) / baseline
        point["throughput_relative_to_craftext_full"] = round(relative, 6)
        point["throughput_degradation_percent"] = round((1.0 - relative) * 100.0, 3)


def benchmark_point(path: Path, batch_size: int, horizon: int, repeats: int) -> BenchmarkPoint:
    """Compile and measure one environment, batch-size and horizon combination.

    :param path: Validated experiment configuration.
    :param batch_size: Number of independently stepped environments.
    :param horizon: Number of ``lax.scan`` steps per measured rollout.
    :param repeats: Number of steady-state executions; must be positive.
    :returns: JSON-compatible point with raw timings and summary statistics.
    """
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    # Deliberately local: the matrix dispatcher must not initialize a JAX runtime.
    import jax
    import jax.numpy as jnp

    from tunix_craftext.config import load_mvp_config
    from tunix_craftext.random_policy import sample_masked_actions
    from tunix_craftext.runtime import build_craftext_runtime

    config = load_mvp_config(path)
    config = replace(
        config,
        environment=replace(config.environment, batch_size=batch_size, horizon=horizon),
    )
    runtime = build_craftext_runtime(config)

    def rollout() -> jax.Array:
        reset = jax.vmap(runtime.adapter.reset)(
            jax.random.split(jax.random.PRNGKey(config.run.seed), batch_size)
        )
        state = reset.state
        action_mask = jnp.broadcast_to(reset.action_mask, (batch_size, runtime.action_count))
        keys = jax.random.split(jax.random.PRNGKey(101), horizon * batch_size * 2).reshape(
            horizon, 2, batch_size, 2
        )

        def scan_step(
            carry: tuple[object, jax.Array], step_keys: jax.Array
        ) -> tuple[tuple[object, jax.Array], jax.Array]:
            current_state, current_mask = carry
            action = sample_masked_actions(step_keys[0], current_mask)
            step = jax.vmap(runtime.adapter.step)(step_keys[1], current_state, action)
            return (step.state, step.action_mask), step.reward

        _, rewards = jax.lax.scan(scan_step, (state, action_mask), keys)
        return rewards

    compiled = jax.jit(rollout)
    compile_started = time.perf_counter()
    jax.block_until_ready(compiled())
    compile_ms = (time.perf_counter() - compile_started) * 1_000

    samples_ms: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        jax.block_until_ready(compiled())
        samples_ms.append((time.perf_counter() - started) * 1_000)

    summary = summarize_samples(samples_ms)
    median_ms = summary["steady_state_median_ms"]
    return {
        "status": "ok",
        "variant": config.run.name,
        "config": str(path),
        "config_sha256": sha256(path.read_bytes()).hexdigest(),
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "batch_size": batch_size,
        "horizon": horizon,
        "warmup_runs": 1,
        "repeats": repeats,
        "compile_and_first_execution_ms": round(compile_ms, 6),
        "samples_ms": [round(sample, 6) for sample in samples_ms],
        **summary,
        "env_steps_per_second_median": round(batch_size * horizon / (median_ms / 1_000), 6),
    }


def git_revision() -> str:
    """Return the current revision without making benchmark recording depend on Git."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unversioned"


def benchmark_payload(points: list[BenchmarkPoint]) -> dict[str, object]:
    """Build provenance common to the whole benchmark matrix artifact."""
    return {
        "schema": SCHEMA,
        "timestamp": datetime.now(UTC).isoformat(),
        "git_revision": git_revision(),
        "git_dirty": bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                text=True,
                capture_output=True,
                check=False,
            ).stdout.strip()
        ),
        "jax_version": version("jax"),
        "hardware": {
            "platform": platform.platform(),
            "backend": "recorded per point",
            "devices": [],
        },
        "notes": {
            "compile_metric": "first JIT invocation, including first execution",
            "steady_metric": "blocking repeated compiled rollouts; median is throughput basis",
            "memory_peak_bytes": None,
            "host_device_bytes": None,
        },
        "points": points,
    }


def write_checkpoint(output: Path, payload: dict[str, object]) -> None:
    """Atomically update the matrix artifact after every attempted point."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def isolated_point(
    config: Path,
    batch_size: int,
    horizon: int,
    repeats: int,
    point_output: Path,
) -> BenchmarkPoint:
    """Run one JIT cell in a child process and return a structured result.

    A native accelerator/compiler failure can terminate Python without raising an
    exception. Isolating each shape protects the remaining matrix and makes that
    failure visible in the artifact rather than silently truncating it.

    :param config: Benchmark configuration for the child process.
    :param batch_size: Environment batch size for the cell.
    :param horizon: Scan horizon for the cell.
    :param repeats: Number of steady-state samples for the cell.
    :param point_output: Temporary JSON destination written by a successful child.
    :returns: Successful point payload or a structured failed record.
    """
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--_point",
        "--config",
        str(config),
        "--batch-size",
        str(batch_size),
        "--horizon",
        str(horizon),
        "--repeats",
        str(repeats),
        "--point-output",
        str(point_output),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    try:
        if completed.returncode == 0 and point_output.is_file():
            payload = json.loads(point_output.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                return payload
        detail = completed.stderr.strip() or completed.stdout.strip()
        return {
            "status": "failed",
            "variant": config.stem,
            "config": str(config),
            "batch_size": batch_size,
            "horizon": horizon,
            "error": f"child exit {completed.returncode}: {detail[-1_000:]}",
        }
    finally:
        point_output.unlink(missing_ok=True)


def main() -> None:
    """Parse the matrix CLI and record every point, including recoverable failures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", type=Path)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 2, 8, 32])
    parser.add_argument("--horizons", nargs="+", type=int, default=[8, 32, 128, 512])
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--_point", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--config", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--batch-size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--horizon", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--point-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._point:
        if (
            args.config is None
            or args.batch_size is None
            or args.horizon is None
            or args.point_output is None
        ):
            parser.error("--_point requires config, batch-size, horizon and point-output")
        point = benchmark_point(args.config, args.batch_size, args.horizon, args.repeats)
        args.point_output.write_text(json.dumps(point) + "\n", encoding="utf-8")
        return

    if not args.configs or args.output is None:
        parser.error("--configs and --output are required for a matrix run")

    points: list[BenchmarkPoint] = []
    payload = benchmark_payload(points)
    for config in args.configs:
        for batch_size in args.batch_sizes:
            for horizon in args.horizons:
                print(
                    f"[{len(points) + 1}] {config.stem} B={batch_size} T={horizon}",
                    flush=True,
                )
                temporary_point = args.output.with_suffix(f".point-{len(points):03d}.json")
                point = isolated_point(config, batch_size, horizon, args.repeats, temporary_point)
                points.append(point)
                add_baseline_comparisons(points)
                write_checkpoint(args.output, payload)
                print(f"  → {point['status']}", flush=True)


if __name__ == "__main__":
    main()
