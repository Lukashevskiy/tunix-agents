"""Tests for reproducible text-episode artifact summaries without a model dependency."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tunix_craftext.artifacts.replay import ReplayArtifact, ReplayStep

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "run_text_episode", ROOT / "scripts" / "run_text_episode.py"
)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)


def test_episode_metrics_retains_invalid_action_and_token_evidence() -> None:
    artifact = ReplayArtifact(
        "configs/mvp/qwen_craftext.yaml",
        "abc",
        "tunix-single-device:Qwen",
        (
            ReplayStep(
                0,
                "prompt",
                "completion",
                0,
                "NOOP",
                1.5,
                False,
                invalid_format=1,
                fallback_used=True,
                token_logprobs=(-1.0, -3.0),
                token_ids=(10, 11),
                prompt_token_ids=(1, 2, 3),
            ),
        ),
    )

    metrics = runner.episode_metrics(artifact)

    assert metrics["steps"] == 1
    assert metrics["reward_sum"] == 1.5
    assert metrics["fallback_count"] == metrics["invalid_format_count"] == 1
    assert metrics["generated_token_count"] == 2
    assert metrics["prompt_token_count"] == 3
    assert metrics["mean_token_logprob"] == -2.0


def test_parse_args_exposes_explicit_artifact_paths() -> None:
    args = runner.parse_args(["--horizon", "2", "--cache-size", "4096"])

    assert args.horizon == 2
    assert args.cache_size == 4096
    assert args.config == Path("configs/mvp/qwen_craftext.yaml")
