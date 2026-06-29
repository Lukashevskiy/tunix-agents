"""Task sampling from CrafText scenario instructions.

Agentic training should not invent a parallel task list when CrafText already
ships scenario instructions.  This module converts a runtime adapter's
instruction rows into serializable Tunix task batches containing both the text
goal and the explicit ``instruction_index`` used by the environment reset.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .runtime import CrafTextRuntime

TaskSamplingMode = Literal["cycle", "fixed", "random"]


@dataclass(frozen=True)
class CrafTextTaskSampler:
    """Sample agentic tasks from CrafText/CagedCrafText instruction rows.

    :param instructions: Scenario instructions exposed by the CrafText adapter.
    :param horizon: Episode horizon attached to every task.
    :param mode: ``cycle`` for deterministic coverage, ``fixed`` for one row, or
        ``random`` for seeded sampling from all instructions.
    :param fixed_instruction_index: Instruction used by ``fixed`` mode and as
        the first row for deterministic validation.
    :param goal_prefix: Optional policy hint prepended to every instruction.
    """

    instructions: tuple[str, ...]
    horizon: int
    mode: TaskSamplingMode = "cycle"
    fixed_instruction_index: int = 0
    goal_prefix: str = ""

    def __post_init__(self) -> None:
        """Validate that sampled tasks can be serialized into Tunix batches."""
        if not self.instructions or any(
            not instruction.strip() for instruction in self.instructions
        ):
            raise ValueError("instructions must contain non-empty CrafText tasks")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.mode not in {"cycle", "fixed", "random"}:
            raise ValueError("unsupported task sampling mode")
        if not 0 <= self.fixed_instruction_index < len(self.instructions):
            raise ValueError("fixed_instruction_index must reference one instruction")

    @classmethod
    def from_runtime(
        cls,
        runtime: CrafTextRuntime,
        *,
        horizon: int,
        mode: TaskSamplingMode = "cycle",
        fixed_instruction_index: int = 0,
        goal_prefix: str = "",
    ) -> CrafTextTaskSampler:
        """Create a sampler from a built CrafText runtime adapter."""
        instructions = getattr(runtime.adapter, "instructions", ())
        if not instructions:
            raise ValueError("runtime adapter does not expose CrafText instructions")
        return cls(
            tuple(str(instruction) for instruction in instructions),
            horizon=horizon,
            mode=mode,
            fixed_instruction_index=fixed_instruction_index,
            goal_prefix=goal_prefix,
        )

    def task_at(self, index: int) -> tuple[str, int]:
        """Return ``(goal, instruction_index)`` for a deterministic task index."""
        if self.mode == "fixed":
            instruction_index = self.fixed_instruction_index
        elif self.mode == "cycle":
            instruction_index = index % len(self.instructions)
        else:
            raise ValueError("task_at is deterministic only for cycle/fixed modes")
        return self._goal(instruction_index), instruction_index

    def batches(
        self,
        *,
        seed: int,
        batch_size: int,
        count: int,
    ) -> Iterator[dict[str, object]]:
        """Yield serializable task batches consumed by Tunix Agentic learners."""
        if batch_size <= 0 or count <= 0:
            raise ValueError("batch_size and count must be positive")
        rng = np.random.default_rng(seed)
        for batch_index in range(count):
            start = batch_index * batch_size
            if self.mode == "random":
                instruction_indices = rng.integers(
                    0, len(self.instructions), size=batch_size, dtype=np.int32
                )
            else:
                instruction_indices = np.asarray(
                    [self.task_at(start + row)[1] for row in range(batch_size)],
                    dtype=np.int32,
                )
            goals = [self._goal(int(index)) for index in instruction_indices]
            seeds = np.arange(seed + start, seed + start + batch_size, dtype=np.int32)
            horizons = np.full(batch_size, self.horizon, dtype=np.int32)
            yield {
                "goal": goals,
                "seed": seeds,
                "horizon": horizons,
                "instruction_index": instruction_indices,
            }

    def validation_tasks(
        self, *, seed: int, limit: int | None = None
    ) -> tuple[dict[str, object], ...]:
        """Return deterministic one-row validation tasks from the instruction list."""
        size = len(self.instructions) if limit is None else min(limit, len(self.instructions))
        return tuple(
            {
                "goal": self._goal(index),
                "seed": seed + index,
                "horizon": self.horizon,
                "instruction_index": index,
            }
            for index in range(size)
        )

    def _goal(self, instruction_index: int) -> str:
        instruction = self.instructions[instruction_index].strip()
        if not self.goal_prefix.strip():
            return instruction
        return f"{self.goal_prefix.strip()}\nCrafText task: {instruction}"


def task_batches_from_craftext(
    sampler: CrafTextTaskSampler,
    *,
    seed: int,
    batch_size: int,
    count: int,
) -> Iterator[dict[str, object]]:
    """Compatibility helper mirroring the old ``task_batches`` script function."""
    yield from sampler.batches(seed=seed, batch_size=batch_size, count=count)
