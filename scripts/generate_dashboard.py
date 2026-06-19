#!/usr/bin/env python3
"""Generate MkDocs dashboard pages from the repository's current evidence.

This is deliberately stdlib-only: `make docs` works in CI and developer machines without
an application server. Generated pages are ignored; their source of truth is Git, the
checkboxes in docs/plan.md, docs/project_status.json, and artifacts/benchmarks/*.json.
"""

from __future__ import annotations

import json
import re
import subprocess
from html import escape
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
GENERATED = DOCS / "_generated"
PLAN = DOCS / "plan.md"
STATUS = DOCS / "project_status.json"
BENCHMARKS = ROOT / "artifacts" / "benchmarks"
CHECKBOX = re.compile(r"^\s*- \[([ xX~])\]\s+(.*)$", re.MULTILINE)


def command(*args: str) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def plan_progress() -> tuple[int, int, list[tuple[str, int, int]]]:
    """Return overall and per-phase checkbox progress from the canonical roadmap."""
    text = PLAN.read_text(encoding="utf-8")
    sections = re.split(r"^##\s+", text, flags=re.MULTILINE)
    breakdown: list[tuple[str, int, int]] = []
    completed = total = 0
    for section in sections[1:]:
        title, _, body = section.partition("\n")
        checks = CHECKBOX.findall(body)
        if checks:
            done = sum(mark.lower() == "x" for mark, _ in checks)
            count = len(checks)
            breakdown.append((title.strip(), done, count))
            completed += done
            total += count
    return completed, total, breakdown


def plan_cards() -> list[tuple[str, str, str]]:
    """Return roadmap cards as ``(phase, status, title)`` for the generated Kanban."""
    text = PLAN.read_text(encoding="utf-8")
    sections = re.split(r"^##\s+", text, flags=re.MULTILINE)
    cards: list[tuple[str, str, str]] = []
    markers = {"x": "done", "~": "active", " ": "planned"}
    for section in sections[1:]:
        title, _, body = section.partition("\n")
        lines = body.splitlines()
        index = 0
        while index < len(lines):
            match = re.match(r"^\s*- \[([ xX~])\]\s+(.*)$", lines[index])
            if match is None:
                index += 1
                continue
            mark, task = match.groups()
            continuation = [task.strip()]
            index += 1
            while index < len(lines) and lines[index].startswith(("  ", "\t")):
                continuation.append(lines[index].strip())
                index += 1
            cards.append((title.strip(), markers[mark.lower()], " ".join(continuation)))
    return cards


def git_metadata() -> dict[str, Any]:
    revision = command("git", "rev-parse", "HEAD") or "unversioned"
    dirty = bool(command("git", "status", "--porcelain"))
    return {
        "revision": revision,
        "short": revision[:8],
        "subject": command("git", "log", "-1", "--format=%s") or "Нет коммитов",
        "author": command("git", "log", "-1", "--format=%an") or "—",
        "date": command("git", "log", "-1", "--format=%aI") or "—",
        "dirty": "да" if dirty else "нет",
        "changed": command("git", "show", "--format=", "--shortstat", "-1") or "Нет данных",
    }


def recent_commits(limit: int = 8) -> list[tuple[str, str, str, str]]:
    raw = command("git", "log", f"-{limit}", "--format=%h%x1f%s%x1f%an%x1f%aI")
    return [tuple(line.split("\x1f", 3)) for line in raw.splitlines() if line]


def benchmark_records() -> list[dict[str, Any]]:
    """Read a small, documented JSON schema and gracefully retain unfamiliar records."""
    records: list[dict[str, Any]] = []
    if not BENCHMARKS.exists():
        return records
    for path in sorted(BENCHMARKS.rglob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        entries: Iterable[Any] = payload if isinstance(payload, list) else [payload]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            metrics = entry.get("metrics", {})
            records.append(
                {
                    "path": path.relative_to(ROOT),
                    "name": entry.get("name", path.stem),
                    "timestamp": entry.get("timestamp", entry.get("created_at", "—")),
                    "commit": entry.get("git_revision", entry.get("commit", "—")),
                    "hardware": entry.get("hardware", entry.get("device", "—")),
                    "metrics": metrics if isinstance(metrics, dict) else {},
                }
            )
    return records


def metric_summary(record: dict[str, Any]) -> str:
    metrics = record["metrics"]
    if not metrics:
        return "—"
    return ", ".join(f"{key}: {value}" for key, value in list(metrics.items())[:4])


def capabilities() -> list[dict[str, str]]:
    payload = json.loads(STATUS.read_text(encoding="utf-8"))
    return payload["capabilities"]


def write_page(name: str, body: str) -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / name).write_text(body.strip() + "\n", encoding="utf-8")


def mermaid_label(value: str, limit: int = 86) -> str:
    """Return a concise Mermaid-safe node label while retaining the task's meaning."""
    normalized = " ".join(value.replace('"', "'").split())
    return normalized if len(normalized) <= limit else f"{normalized[: limit - 1]}…"


def generate() -> None:
    done, total, phases = plan_progress()
    percent = 0 if not total else round(100 * done / total, 1)
    git = git_metadata()
    caps = capabilities()
    ready = [item for item in caps if item["status"] == "ready"]
    benchmarks = benchmark_records()
    built_at = datetime.now(timezone.utc).isoformat()

    phase_rows = "\n".join(
        f"| {markdown_escape(title)} | {finished}/{count} | "
        f"{round(100 * finished / count) if count else 0}% |"
        for title, finished, count in phases
    )
    capability_rows = "\n".join(
        f"| {markdown_escape(item['name'])} | {'Готово' if item['status'] == 'ready' else 'Запланировано'} | "
        f"{markdown_escape(item['description'])} |"
        for item in caps
    )
    dashboard = f"""
# Dashboard проекта

_Сгенерировано автоматически: `{built_at}`. Источники: Git, `docs/plan.md`,
`docs/project_status.json`, `artifacts/benchmarks/*.json`._

## Состояние сейчас

| Показатель | Значение |
| --- | --- |
| Прогресс roadmap | **{done}/{total} задач ({percent}%)** |
| Последний commit | `{git['short']}` — {markdown_escape(git['subject'])} |
| Автор / дата | {markdown_escape(git['author'])} / `{git['date']}` |
| Working tree dirty | {git['dirty']} |
| Benchmark records | {len(benchmarks)} |
| Готовые возможности | {len(ready)} |

## Продвижение по плану

| Этап | Готово | Прогресс |
| --- | ---: | ---: |
{phase_rows}

Полный и редактируемый roadmap — в [плане реализации](../plan.md).

## Что уже можно делать

| Возможность | Статус | Что это даёт |
| --- | --- | --- |
{capability_rows}

## Последнее изменение

```text
{git['changed']}
```

Смотрите [историю улучшений](changelog.md) и [результаты benchmark](benchmarks.md).
"""
    write_page("dashboard.md", dashboard)

    cards = plan_cards()
    phases = []
    for phase, _, _ in cards:
        if phase not in phases:
            phases.append(phase)
    lane_names = (("done", "Сделано"), ("active", "В текущей реализации"), ("planned", "Запланировано"))
    boards = []
    for phase in phases:
        phase_cards = [card for card in cards if card[0] == phase]
        lanes = []
        for status, label in lane_names:
            card_html = "".join(
                f'<article class="kanban-card">{escape(task)}</article>'
                for _, item_status, task in phase_cards
                if item_status == status
            )
            lane_contents = card_html or '<p class="kanban-empty">Нет карточек</p>'
            lanes.append(
                f'<section class="kanban-lane kanban-lane--{status}"><h3>{label}</h3>'
                f"{lane_contents}</section>"
            )
        boards.append(f"## {escape(phase)}\n\n<div class=\"kanban-board\">{''.join(lanes)}</div>")
    write_page(
        "kanban.md",
        f"""
# Тематический Kanban

_Автогенерация: `{built_at}` из `docs/plan.md`. Меняйте статус задачи в roadmap:_
`[x]` — сделано, `[~]` — текущая реализация, `[ ]` — запланировано.

{chr(10).join(boards)}
""",
    )

    status_counts = {status: sum(1 for _, item_status, _ in cards if item_status == status) for status, _ in lane_names}
    graph_lines = ["flowchart LR"]
    status_classes = {"done": "done", "active": "active", "planned": "planned"}
    for phase_index, phase in enumerate(phases):
        phase_id = f"phase{phase_index}"
        graph_lines.extend([f'  subgraph {phase_id}["{mermaid_label(phase, 50)}"]', "    direction TB"])
        for task_index, (_, status, task) in enumerate(card for card in cards if card[0] == phase):
            task_id = f"task{phase_index}_{task_index}"
            graph_lines.append(f'    {task_id}["{mermaid_label(task)}"]:::{status_classes[status]}')
        graph_lines.append("  end")
    graph_lines.extend(
        [
            "  classDef done fill:#238636,stroke:#3fb950,color:#fff;",
            "  classDef active fill:#9e6a03,stroke:#d29922,color:#fff;",
            "  classDef planned fill:#1f6feb,stroke:#58a6ff,color:#fff;",
        ]
    )
    write_page(
        "task-graph.md",
        f"""
# Граф задач и объёма работы

_Автогенерация: `{built_at}` из `docs/plan.md`._

| Статус | Задач |
| --- | ---: |
| Сделано | {status_counts['done']} |
| В текущей реализации | {status_counts['active']} |
| Запланировано | {status_counts['planned']} |

```mermaid
{chr(10).join(graph_lines)}
```

Каждый кластер — тематический модуль/этап, каждый узел — отдельная задача. Цвета совпадают с
Kanban: зелёный — готово, янтарный — в текущей реализации, синий — запланировано. Сейчас граф
показывает объём и принадлежность задач к модулям; dependency-стрелки появятся из
`docs/tasks.json`, когда будет добавлен интерактивный редактор.
""",
    )

    commit_rows = "\n".join(
        f"| `{sha}` | {markdown_escape(subject)} | {markdown_escape(author)} | `{date}` |"
        for sha, subject, author, date in recent_commits()
    ) or "| — | История пока пуста | — | — |"
    write_page(
        "changelog.md",
        f"""
# Последние улучшения

_Автогенерация: `{built_at}`._

| Commit | Изменение | Автор | Дата |
| --- | --- | --- | --- |
{commit_rows}

Детали последнего commit:

```text
{git['changed']}
```
""",
    )

    benchmark_rows = "\n".join(
        f"| {markdown_escape(item['name'])} | `{item['timestamp']}` | `{item['commit']}` | "
        f"{markdown_escape(item['hardware'])} | {markdown_escape(metric_summary(item))} | "
        f"`{item['path']}` |"
        for item in benchmarks
    )
    empty = (
        "\n> Пока нет benchmark artifacts. После первого запуска положите JSON в "
        "`artifacts/benchmarks/`; следующая сборка автоматически добавит его сюда.\n"
    )
    write_page(
        "benchmarks.md",
        f"""
# Бенчмарки

_Автогенерация: `{built_at}`. Отображаются все JSON-файлы из `artifacts/benchmarks/`._
{empty if not benchmark_rows else ''}
| Сценарий | Время | Commit | Hardware | Метрики | Артефакт |
| --- | --- | --- | --- | --- | --- |
{benchmark_rows}

## Формат artifact

```json
{{
  "name": "craftext-rollout-256x128",
  "timestamp": "2026-06-20T12:00:00Z",
  "git_revision": "<commit>",
  "hardware": "device description",
  "metrics": {{"env_steps_per_second": 0.0, "compile_seconds": 0.0}}
}}
```
""",
    )


if __name__ == "__main__":
    generate()
