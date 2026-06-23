"""Tests for inspectable text-pipeline performance evidence."""

from __future__ import annotations

from scripts.benchmark_text_pipeline import parse_args, summarize_samples


def test_phase_summary_retains_raw_count_and_percentiles() -> None:
    """A phase summary makes a small trace independently auditable."""
    summary = summarize_samples([1.0, 2.0, 5.0])

    assert summary == {
        "count": 3,
        "mean_ms": 2.666667,
        "median_ms": 2.0,
        "p95_ms": 4.7,
        "min_ms": 1.0,
        "max_ms": 5.0,
    }


def test_benchmark_can_isolate_native_model_runs() -> None:
    """The standard CLI exposes a process-isolated repeat mode."""
    args = parse_args(["--isolate-runs", "--repeats", "10"])

    assert args.isolate_runs is True
    assert args.repeats == 10
