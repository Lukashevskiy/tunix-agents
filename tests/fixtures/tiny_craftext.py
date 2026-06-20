"""Deterministic in-memory CrafText-shaped environments for adapter golden tests."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TinyState:
    """Opaque stand-in state carrying only a deterministic timestep."""

    timestep: int


class TinyCrafText:
    """CrafText-shaped reset/step API with action masks for contract-level tests."""

    def __init__(self, terminal_timestep: int, caged: bool = False) -> None:
        self.terminal_timestep = terminal_timestep
        self.caged = caged

    def reset(self, key: int, params: object) -> tuple[np.ndarray, TinyState]:
        del params
        return np.asarray([key, 0], dtype=np.int32), TinyState(timestep=0)

    def step(
        self, key: int, state: TinyState, action: int, params: object
    ) -> tuple[np.ndarray, TinyState, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        del key, params
        next_timestep = state.timestep + 1
        reward = np.asarray(float(action + (1 if self.caged else 0)), dtype=np.float32)
        done = np.asarray(next_timestep >= self.terminal_timestep)
        mask = np.asarray([True, action % 2 == 0, not self.caged], dtype=bool)
        return (
            np.asarray([next_timestep, action], dtype=np.int32),
            TinyState(timestep=next_timestep),
            reward,
            done,
            {"action_mask": mask},
        )
