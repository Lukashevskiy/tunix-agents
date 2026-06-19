import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("dashboard", ROOT / "scripts" / "generate_dashboard.py")
dashboard = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(dashboard)


def test_plan_progress_counts_completed_and_pending_items() -> None:
    completed, total, phases = dashboard.plan_progress()

    assert completed == 6
    assert total == 33
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
