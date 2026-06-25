"""Contract tests for the explicit Agentic GRPO runner without model allocation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_agentic_grpo", ROOT / "scripts/run_agentic_grpo.py"
)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def test_task_batches_are_deterministic_and_groupable() -> None:
    batches = list(
        runner.task_batches(goal="collect wood", seed=7, batch_size=2, count=2, horizon=3)
    )

    assert [batch["seed"].tolist() for batch in batches] == [[7, 8], [9, 10]]
    assert all(batch["goal"] == ["collect wood", "collect wood"] for batch in batches)
    assert all(batch["horizon"].tolist() == [3, 3] for batch in batches)


def test_runner_arguments_keep_model_and_topology_explicit() -> None:
    args = runner.parse_args(["--max-steps", "2", "--num-generations", "3", "--dry-run"])

    assert args.max_steps == 2
    assert args.num_generations == 3
    assert args.max_prompt_length == 1024
    assert args.kv_cache_size == 2048
    assert not args.allow_cpu_smoke
    assert args.snapshot == Path("artifacts/models/qwen25-05b-instruct")
    assert args.dry_run


def test_runner_exposes_scripted_grpo_smoke_without_model_assets() -> None:
    args = runner.parse_args(["--scripted-smoke", "--scripted-output", "out.json"])

    assert args.scripted_smoke
    assert args.scripted_output == Path("out.json")
