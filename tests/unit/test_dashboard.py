import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "dashboard", ROOT / "scripts" / "generate_dashboard.py"
)
dashboard = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(dashboard)


def test_plan_progress_counts_completed_and_pending_items() -> None:
    completed, total, phases = dashboard.plan_progress()

    assert completed == 8
    assert total == 34
    assert phases[0][0].startswith("0.")


def test_dashboard_generation_writes_current_project_pages() -> None:
    dashboard.generate()

    rendered = (ROOT / "docs" / "_generated" / "dashboard.md").read_text(encoding="utf-8")
    assert "Прогресс roadmap" in rendered
    assert "Что уже можно делать" in rendered

    kanban = (ROOT / "docs" / "_generated" / "kanban.md").read_text(encoding="utf-8")
    assert "kanban-lane--active" in kanban
    assert "CrafTextAdapter.reset/step" in kanban
    assert "action-mask" in kanban

    graph = (ROOT / "docs" / "_generated" / "task-graph.md").read_text(encoding="utf-8")
    assert 'id="task-graph-data"' in graph
    assert '"type": "task"' in graph


def test_dashboard_reads_v2_environment_benchmark(monkeypatch, tmp_path: Path) -> None:
    """Dashboard exposes median/p95 and full-preset comparison from the v2 schema."""
    artifact = tmp_path / "environment.json"
    artifact.write_text(
        json.dumps(
            {
                "schema": "tunix-craftext.environment-benchmark/v2",
                "timestamp": "2026-01-01T00:00:00Z",
                "git_revision": "abc",
                "hardware": {"platform": "test", "backend": "cpu"},
                "points": [
                    {
                        "status": "ok",
                        "variant": "craftext-full",
                        "batch_size": 2,
                        "horizon": 8,
                        "compile_and_first_execution_ms": 10.0,
                        "steady_state_median_ms": 2.0,
                        "steady_state_p95_ms": 3.0,
                        "env_steps_per_second_median": 8_000.0,
                        "throughput_relative_to_craftext_full": 1.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(dashboard, "BENCHMARKS", tmp_path)

    records = dashboard.benchmark_records()

    assert records[0]["metrics"]["median_ms"] == 2.0
    assert records[0]["metrics"]["p95_ms"] == 3.0
    assert records[0]["metrics"]["vs_full"] == 1.0
