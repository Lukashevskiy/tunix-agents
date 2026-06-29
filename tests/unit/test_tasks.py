"""Tests for sampling Agentic tasks from CrafText scenario instructions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from tunix_craftext.env.tasks import CrafTextTaskSampler, task_batches_from_craftext


@dataclass(frozen=True)
class _Runtime:
    adapter: object


@dataclass(frozen=True)
class _Adapter:
    instructions: tuple[str, ...]


def test_craftext_task_sampler_cycles_goals_and_instruction_indices() -> None:
    sampler = CrafTextTaskSampler(
        ("collect wood", "avoid enemy"),
        horizon=5,
        goal_prefix="Obey the CrafText task.",
    )

    [batch] = list(sampler.batches(seed=7, batch_size=3, count=1))

    assert batch["goal"] == [
        "Obey the CrafText task.\nCrafText task: collect wood",
        "Obey the CrafText task.\nCrafText task: avoid enemy",
        "Obey the CrafText task.\nCrafText task: collect wood",
    ]
    np.testing.assert_array_equal(batch["seed"], [7, 8, 9])
    np.testing.assert_array_equal(batch["horizon"], [5, 5, 5])
    np.testing.assert_array_equal(batch["instruction_index"], [0, 1, 0])


def test_craftext_task_sampler_random_mode_is_seeded() -> None:
    sampler = CrafTextTaskSampler(("a", "b", "c"), horizon=2, mode="random")

    first = list(sampler.batches(seed=3, batch_size=4, count=2))
    second = list(sampler.batches(seed=3, batch_size=4, count=2))

    for left, right in zip(first, second, strict=True):
        assert left["goal"] == right["goal"]
        np.testing.assert_array_equal(left["instruction_index"], right["instruction_index"])


def test_craftext_task_sampler_fixed_mode_and_validation_tasks() -> None:
    sampler = CrafTextTaskSampler(
        ("collect wood", "avoid enemy", "sleep"),
        horizon=8,
        mode="fixed",
        fixed_instruction_index=1,
    )

    [batch] = list(task_batches_from_craftext(sampler, seed=10, batch_size=2, count=1))
    validation = sampler.validation_tasks(seed=100, limit=2)

    assert batch["goal"] == ["avoid enemy", "avoid enemy"]
    np.testing.assert_array_equal(batch["instruction_index"], [1, 1])
    assert validation == (
        {"goal": "collect wood", "seed": 100, "horizon": 8, "instruction_index": 0},
        {"goal": "avoid enemy", "seed": 101, "horizon": 8, "instruction_index": 1},
    )


def test_craftext_task_sampler_from_runtime_requires_adapter_instructions() -> None:
    sampler = CrafTextTaskSampler.from_runtime(
        _Runtime(_Adapter(("collect wood",))),
        horizon=4,
    )

    assert sampler.task_at(0) == ("collect wood", 0)
    with pytest.raises(ValueError, match="does not expose"):
        CrafTextTaskSampler.from_runtime(_Runtime(object()), horizon=4)
