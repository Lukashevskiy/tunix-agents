#!/usr/bin/env python3
"""Generate MkDocs dashboard pages from the repository's current evidence.

This is deliberately stdlib-only: `make docs` works in CI and developer machines without
an application server. Generated pages are ignored; their source of truth is Git, the
checkboxes in docs/plan.md, docs/project_status.json, and artifacts/benchmarks/*.json.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
GENERATED = DOCS / "_generated"
PLAN = DOCS / "plan.md"
STATUS = DOCS / "project_status.json"
BENCHMARKS = ROOT / "artifacts" / "benchmarks"
TEXT_EPISODE_METRICS = ROOT / "artifacts" / "metrics"
CHECKBOX = re.compile(r"^\s*- \[([ xX~])\]\s+(.*)$", re.MULTILINE)
API_MODULES = (
    "tunix_craftext.adapters.craftext",
    "tunix_craftext.prompts",
    "tunix_craftext.text_policy",
    "tunix_craftext.llm",
    "tunix_craftext.batched_rollout",
    "tunix_craftext.hybrid_rollout",
    "tunix_craftext.replay",
    "tunix_craftext.text_trajectory",
    "tunix_craftext.algorithms",
    "tunix_craftext.algorithm_registry",
    "tunix_craftext.config",
    "tunix_craftext.runtime",
    "tunix_craftext.profiling",
    "tunix_craftext.checkpoints",
    "tunix_craftext.tunix_adapter",
    "tunix_craftext.tunix_actor",
    "tunix_craftext.tunix_topology",
    "tunix_craftext.rlcluster_workload",
)


def command(*args: str) -> str:
    try:
        return subprocess.check_output(args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def markdown_escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def display_artifact_path(path: Path) -> Path | str:
    """Return a repository-relative artifact name, retaining external test paths safely."""
    try:
        return path.relative_to(ROOT)
    except ValueError:
        return path


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
    for path in sorted(
        BENCHMARKS.rglob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True
    ):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict) and isinstance(payload.get("benchmarks"), list):
            machine = payload.get("machine_info", {})
            commit = payload.get("commit_info", {})
            for benchmark in payload["benchmarks"]:
                if not isinstance(benchmark, dict):
                    continue
                stats = benchmark.get("stats", {})
                if not isinstance(stats, dict):
                    stats = {}
                mean = stats.get("mean")
                metrics = {
                    "mean_ms": round(float(mean) * 1000, 4)
                    if isinstance(mean, (int, float))
                    else "—",
                    "median_ms": round(float(stats["median"]) * 1000, 4)
                    if isinstance(stats.get("median"), (int, float))
                    else "—",
                    "ops": round(1 / float(mean), 2)
                    if isinstance(mean, (int, float)) and mean
                    else "—",
                }
                records.append(
                    {
                        "path": display_artifact_path(path),
                        "name": benchmark.get("name", path.stem),
                        "timestamp": payload.get("datetime", "—"),
                        "commit": commit.get("id", "—") if isinstance(commit, dict) else "—",
                        "hardware": (
                            machine.get("cpu", {}).get("brand_raw")
                            or machine.get("cpu", {}).get("arch_string_raw")
                            or machine.get("machine", "—")
                        )
                        if isinstance(machine, dict)
                        else "—",
                        "metrics": metrics,
                    }
                )
        elif isinstance(payload, dict) and payload.get("schema") in {
            "tunix-craftext.environment-benchmark/v1",
            "tunix-craftext.environment-benchmark/v2",
        }:
            hardware = payload.get("hardware", "—")
            if isinstance(hardware, dict):
                hardware = f"{hardware.get('platform', '—')} / {hardware.get('backend', '—')}"
            for point in payload.get("points", []):
                if not isinstance(point, dict):
                    continue
                is_v2 = payload.get("schema") == "tunix-craftext.environment-benchmark/v2"
                metrics: dict[str, object]
                if point.get("status") == "failed":
                    metrics = {"status": "failed", "error": point.get("error", "unknown")}
                elif is_v2:
                    metrics = {
                        "compile_ms": point.get("compile_and_first_execution_ms", "—"),
                        "median_ms": point.get("steady_state_median_ms", "—"),
                        "p95_ms": point.get("steady_state_p95_ms", "—"),
                        "env_steps_per_second": point.get("env_steps_per_second_median", "—"),
                        "vs_full": point.get("throughput_relative_to_craftext_full", "—"),
                    }
                else:
                    metrics = {
                        "mean_ms": point.get("steady_state_ms", "—"),
                        "compile_ms": point.get("compile_ms", "—"),
                        "env_steps_per_second": point.get("env_steps_per_second", "—"),
                    }
                records.append(
                    {
                        "path": display_artifact_path(path),
                        "name": (
                            f"{point.get('variant', 'environment')} "
                            f"B={point.get('batch_size', '—')} T={point.get('horizon', '—')}"
                        ),
                        "timestamp": payload.get("timestamp", "—"),
                        "commit": payload.get("git_revision", payload.get("commit", "—")),
                        "hardware": (
                            f"{hardware} / {point.get('jax_backend', '—')}" if is_v2 else hardware
                        ),
                        "metrics": metrics,
                    }
                )
            continue
        elif isinstance(payload, dict) and payload.get("schema") == (
            "tunix-craftext.text-pipeline-benchmark/v1"
        ):
            phases = payload.get("phase_summaries", {})
            if not isinstance(phases, dict):
                phases = {}
            generation = phases.get("llm_generation_ms", {})
            total = phases.get("decision_total_ms", {})
            if not isinstance(generation, dict):
                generation = {}
            if not isinstance(total, dict):
                total = {}
            records.append(
                {
                    "path": display_artifact_path(path),
                    "name": "Qwen/CrafText synchronous pipeline",
                    "timestamp": payload.get("timestamp", "—"),
                    "commit": payload.get("git_revision", "—"),
                    "hardware": markdown_escape(payload.get("hardware", "—")),
                    "metrics": {
                        "generation_median_ms": generation.get("median_ms", "—"),
                        "generation_p95_ms": generation.get("p95_ms", "—"),
                        "decision_median_ms": total.get("median_ms", "—"),
                        "decision_p95_ms": total.get("p95_ms", "—"),
                    },
                }
            )
            continue
        entries: Iterable[Any] = payload if isinstance(payload, list) else [payload]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            metrics = entry.get("metrics", {})
            records.append(
                {
                    "path": display_artifact_path(path),
                    "name": entry.get("name", path.stem),
                    "timestamp": entry.get("timestamp", entry.get("created_at", "—")),
                    "commit": entry.get("git_revision", entry.get("commit", "—")),
                    "hardware": entry.get("hardware", entry.get("device", "—")),
                    "metrics": metrics if isinstance(metrics, dict) else {},
                }
            )
    return records


def latest_text_episode_metrics() -> dict[str, Any] | None:
    """Return the most recently written versioned Qwen/CrafText episode summary."""
    if not TEXT_EPISODE_METRICS.exists():
        return None
    for path in sorted(
        TEXT_EPISODE_METRICS.rglob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True
    ):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            isinstance(payload, dict)
            and payload.get("schema") == "tunix-craftext.text-episode-metrics/v1"
        ):
            return {"path": display_artifact_path(path), **payload}
    return None


def text_episode_table(record: dict[str, Any] | None) -> str:
    """Render the latest text-episode evidence as a compact dashboard table."""
    if record is None:
        return (
            "> Нет `artifacts/metrics/*` с schema `tunix-craftext.text-episode-metrics/v1`. "
            "Запустите `scripts/run_text_episode.py`."
        )
    rows = (
        ("Время", f"`{record['created_at']}`"),
        ("Steps / reward", f"{record['steps']} / {record['reward_sum']}"),
        (
            "Prompt / generated tokens",
            f"{record.get('prompt_token_count', '—')} / {record['generated_token_count']}",
        ),
        (
            "Fallback",
            str(record["fallback_count"]),
        ),
        (
            "Invalid format / unknown action",
            f"{record['invalid_format_count']} / {record['unknown_action_count']}",
        ),
        ("Artifact", f"`{record['path']}`"),
    )
    return "\n".join(["| Показатель | Значение |", "| --- | --- |"] + [
        f"| {name} | {value} |" for name, value in rows
    ])


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


def first_doc_paragraph(doc: str | None) -> str:
    """Return the first paragraph of a docstring, collapsed for a Markdown table."""
    paragraph = (doc or "").split("\n\n", 1)[0]
    return markdown_escape(" ".join(paragraph.split())) if paragraph else "—"


def _api_module_path(module_name: str) -> Path:
    """Map one configured package module name to its source file strictly under ``src``."""
    package, _, suffix = module_name.partition(".")
    if package != "tunix_craftext" or not suffix:
        raise ValueError(f"API module must be under tunix_craftext: {module_name}")
    return ROOT / "src" / Path(*module_name.split(".")).with_suffix(".py")


def _ast_signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    """Render a lightweight source-level function signature without importing its module."""
    signature = f"({ast.unparse(node.args)})"
    if node.returns is not None:
        signature = f"{signature} -> {ast.unparse(node.returns)}"
    return signature


def public_api_rows(module_name: str) -> str:
    """Render public source-level classes/functions without importing optional dependencies."""
    try:
        source_path = _api_module_path(module_name)
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    except (OSError, SyntaxError, ValueError) as error:
        return f"| `{module_name}` | source failed | — | {markdown_escape(error)} |"
    rows: list[str] = []
    for node in sorted(tree.body, key=lambda value: getattr(value, "name", "")):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        kind = "class" if isinstance(node, ast.ClassDef) else "function"
        signature = "(...)" if kind == "class" else _ast_signature(node)
        rows.append(
            f"| `{module_name}` | {kind} | `{node.name}{signature}` | "
            f"{first_doc_paragraph(ast.get_docstring(node))} |"
        )
    return "\n".join(rows) or f"| `{module_name}` | — | — | Нет public API symbols. |"


def generate_api_reference(built_at: str) -> None:
    """Generate a lightweight MkDocs API reference from Python docstrings."""
    module_sections = "\n".join(
        f"### `{module}`\n\n"
        "| Модуль | Тип | Symbol | Описание |\n"
        "| --- | --- | --- | --- |\n"
        f"{public_api_rows(module)}\n"
        for module in API_MODULES
    )
    write_page(
        "api-reference.md",
        f"""
# Автодока API

_Автогенерация: `{built_at}` из Python docstrings в `src/tunix_craftext`.
Генератор: `scripts/generate_dashboard.py`. Для Sphinx/autodoc HTML также используйте
`make api-docs`._

Эта страница намеренно stdlib-only, чтобы MkDocs site мог получить code reference без
дополнительных plugins. Она перечисляет public classes/functions каждого слоя; подробные
контракты, примеры и типы берутся из docstrings рядом с кодом.

{module_sections}
""",
    )


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
    text_episode = latest_text_episode_metrics()
    text_episode_rows = text_episode_table(text_episode)
    built_at = datetime.now(UTC).isoformat()
    generate_api_reference(built_at)

    phase_rows = "\n".join(
        f"| {markdown_escape(title)} | {finished}/{count} | "
        f"{round(100 * finished / count) if count else 0}% |"
        for title, finished, count in phases
    )
    capability_rows = "\n".join(
        f"| {markdown_escape(item['name'])} | "
        f"{'Готово' if item['status'] == 'ready' else 'Запланировано'} | "
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
| Последний commit | `{git["short"]}` — {markdown_escape(git["subject"])} |
| Автор / дата | {markdown_escape(git["author"])} / `{git["date"]}` |
| Working tree dirty | {git["dirty"]} |
| Benchmark records | {len(benchmarks)} |
| Последний LLM episode | {"есть" if text_episode else "нет"} |
| Готовые возможности | {len(ready)} |

## Продвижение по плану

| Этап | Готово | Прогресс |
| --- | ---: | ---: |
{phase_rows}

Полный и редактируемый roadmap — в [плане реализации](../plan.md).

## Последний Qwen/CrafText episode

{text_episode_rows}

## Что уже можно делать

| Возможность | Статус | Что это даёт |
| --- | --- | --- |
{capability_rows}

## Последнее изменение

```text
{git["changed"]}
```

Смотрите [историю улучшений](changelog.md) и [результаты benchmark](benchmarks.md).
"""
    write_page("dashboard.md", dashboard)

    cards = plan_cards()
    phases = []
    for phase, _, _ in cards:
        if phase not in phases:
            phases.append(phase)
    lane_names = (
        ("done", "Сделано"),
        ("active", "В текущей реализации"),
        ("planned", "Запланировано"),
    )
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
        boards.append(f'## {escape(phase)}\n\n<div class="kanban-board">{"".join(lanes)}</div>')
    write_page(
        "kanban.md",
        f"""
# Тематический Kanban

_Автогенерация: `{built_at}` из `docs/plan.md`. Меняйте статус задачи в roadmap:_
`[x]` — сделано, `[~]` — текущая реализация, `[ ]` — запланировано.

{chr(10).join(boards)}
""",
    )

    status_counts = {
        status: sum(1 for _, item_status, _ in cards if item_status == status)
        for status, _ in lane_names
    }
    palette = {
        "done": {"color": "#238636", "border": "#3fb950", "label": "Сделано"},
        "active": {"color": "#9e6a03", "border": "#d29922", "label": "В текущей реализации"},
        "planned": {"color": "#1f6feb", "border": "#58a6ff", "label": "Запланировано"},
    }
    graph_elements = []
    previous_task = None
    for phase_index, phase in enumerate(phases):
        phase_id = f"phase{phase_index}"
        graph_elements.append({"data": {"id": phase_id, "label": phase, "type": "module"}})
        for task_index, (_, status, task) in enumerate(card for card in cards if card[0] == phase):
            task_id = f"task{phase_index}_{task_index}"
            graph_elements.append(
                {
                    "data": {
                        "id": task_id,
                        "parent": phase_id,
                        "type": "task",
                        "label": mermaid_label(task, 56),
                        "title": task,
                        "description": task,
                        "module": phase,
                        "statusLabel": palette[status]["label"],
                        "color": palette[status]["color"],
                        "border": palette[status]["border"],
                        "previous": previous_task["title"] if previous_task else "",
                    }
                }
            )
            if previous_task:
                graph_elements.append(
                    {
                        "data": {
                            "id": f"order-{previous_task['id']}-{task_id}",
                            "source": previous_task["id"],
                            "target": task_id,
                        }
                    }
                )
            previous_task = {"id": task_id, "title": task}
    graph_json = (
        json.dumps({"elements": graph_elements}, ensure_ascii=False)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    write_page(
        "task-graph.md",
        f"""
# Граф задач и объёма работы

_Автогенерация: `{built_at}` из `docs/plan.md`._

| Статус | Задач |
| --- | ---: |
| Сделано | {status_counts["done"]} |
| В текущей реализации | {status_counts["active"]} |
| Запланировано | {status_counts["planned"]} |

<div class="task-graph-toolbar">
  <button id="task-graph-fit" type="button">Показать весь граф</button>
  <button id="task-graph-layout" type="button">Переложить узлы</button>
  <span>Колесо — zoom · drag background — pan · drag task — переместить ·
  click — карточка задачи</span>
</div>
<div id="task-graph"></div>
<div id="task-graph-detail" class="task-graph-detail">Выберите задачу, чтобы увидеть её
описание и связь.</div>
<script id="task-graph-data" type="application/json">{graph_json}</script>

Каждый узел — конкретная задача, а стрелки показывают текущий **выведенный порядок исполнения**
из roadmap, включая переход между модулями. Это позволяет увидеть критический «хвост» уже сейчас;
после появления `docs/tasks.json` стрелки будут заменены на явно заданные зависимости.
""",
    )

    commit_rows = (
        "\n".join(
            f"| `{sha}` | {markdown_escape(subject)} | {markdown_escape(author)} | `{date}` |"
            for sha, subject, author, date in recent_commits()
        )
        or "| — | История пока пуста | — | — |"
    )
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
{git["changed"]}
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
{empty if not benchmark_rows else ""}
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
