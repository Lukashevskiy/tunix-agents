#!/usr/bin/env python3
"""Synchronize task-derived site views and validate optional structured task dependencies."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set


ROOT = Path(__file__).resolve().parents[4]
PLAN = ROOT / "docs" / "plan.md"
KANBAN = ROOT / "docs" / "_generated" / "kanban.md"
TASKS = ROOT / "docs" / "tasks.json"
CHECKBOX = re.compile(r"^\s*- \[([ xX~])\]\s+", re.MULTILINE)


def task_records() -> Iterable[Dict[str, object]]:
    """Yield structured task records when the local editor has created its registry."""
    if not TASKS.exists():
        return []
    payload = json.loads(TASKS.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), list):
        raise ValueError("docs/tasks.json must contain a tasks list")
    return payload["tasks"]


def validate_dependencies(records: Iterable[Dict[str, object]]) -> int:
    """Reject duplicate IDs, dangling dependencies and directed dependency cycles."""
    tasks = list(records)
    identifiers = [item.get("id") for item in tasks]
    if any(not isinstance(identifier, str) or not identifier for identifier in identifiers):
        raise ValueError("every structured task requires a non-empty string id")
    ids = set(identifiers)
    if len(ids) != len(identifiers):
        raise ValueError("structured task IDs must be unique")
    graph: Dict[str, List[str]] = {}
    for item in tasks:
        task_id = item["id"]
        dependencies = item.get("dependencies", [])
        if not isinstance(dependencies, list) or not all(isinstance(value, str) for value in dependencies):
            raise ValueError(f"dependencies for {task_id} must be a string list")
        unknown = set(dependencies) - ids
        if unknown:
            raise ValueError(f"{task_id} has unknown dependencies: {', '.join(sorted(unknown))}")
        graph[task_id] = dependencies

    visiting: Set[str] = set()
    visited: Set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visiting:
            raise ValueError(f"dependency cycle includes {task_id}")
        if task_id in visited:
            return
        visiting.add(task_id)
        for dependency in graph[task_id]:
            visit(dependency)
        visiting.remove(task_id)
        visited.add(task_id)

    for task_id in graph:
        visit(task_id)
    return len(tasks)


def main() -> None:
    records = task_records()
    dependency_count = validate_dependencies(records)
    subprocess.run(["make", "docs"], cwd=ROOT, check=True)
    roadmap_count = len(CHECKBOX.findall(PLAN.read_text(encoding="utf-8")))
    kanban_count = KANBAN.read_text(encoding="utf-8").count('class="kanban-card"')
    if roadmap_count != kanban_count:
        raise RuntimeError(f"roadmap has {roadmap_count} tasks but Kanban has {kanban_count} cards")
    print(
        json.dumps(
            {
                "roadmap_tasks": roadmap_count,
                "kanban_cards": kanban_count,
                "structured_tasks": dependency_count,
                "dependency_validation": "passed" if TASKS.exists() else "deferred until docs/tasks.json exists",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"task-board sync failed: {error}", file=sys.stderr)
        raise SystemExit(1)
