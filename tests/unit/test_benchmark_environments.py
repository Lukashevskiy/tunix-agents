"""Unit tests for benchmark statistics and cross-variant comparison semantics."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_environments", ROOT / "scripts" / "benchmark_environments.py"
)
benchmark = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(benchmark)


def test_sample_summary_reports_interpolated_p95() -> None:
    """Median and p95 remain independent of sample arrival order."""
    summary = benchmark.summarize_samples([10.0, 40.0, 20.0, 30.0, 50.0])

    assert summary == {
        "steady_state_mean_ms": 30.0,
        "steady_state_median_ms": 30.0,
        "steady_state_p95_ms": 48.0,
        "steady_state_min_ms": 10.0,
        "steady_state_max_ms": 50.0,
    }


def test_sample_summary_rejects_empty_measurement_set() -> None:
    """A benchmark can never silently publish a divide-by-zero result."""
    with pytest.raises(ValueError, match="at least one"):
        benchmark.summarize_samples([])


def test_baseline_comparison_matches_only_same_matrix_cell() -> None:
    """Variant comparison uses the matching batch and horizon baseline only."""
    points = [
        {
            "variant": "craftext-full",
            "batch_size": 2,
            "horizon": 8,
            "env_steps_per_second_median": 100.0,
        },
        {
            "variant": "caged-craftext-full",
            "batch_size": 2,
            "horizon": 8,
            "env_steps_per_second_median": 80.0,
        },
        {
            "variant": "craftext-tiny",
            "batch_size": 2,
            "horizon": 32,
            "env_steps_per_second_median": 200.0,
        },
    ]

    benchmark.add_baseline_comparisons(points)

    assert points[0]["throughput_relative_to_craftext_full"] == 1.0
    assert points[1]["throughput_relative_to_craftext_full"] == 0.8
    assert points[1]["throughput_degradation_percent"] == 20.0
    assert "throughput_relative_to_craftext_full" not in points[2]


def test_isolated_point_records_native_child_failure(monkeypatch, tmp_path: Path) -> None:
    """A killed JIT child becomes a failure record instead of aborting the matrix."""
    monkeypatch.setattr(
        benchmark.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=-9, stderr="killed", stdout=""),
    )

    point = benchmark.isolated_point(
        Path("configs/benchmarks/craftext_full.yaml"),
        batch_size=32,
        horizon=512,
        repeats=20,
        point_output=tmp_path / "missing.json",
    )

    assert point["status"] == "failed"
    assert "child exit -9" in str(point["error"])
