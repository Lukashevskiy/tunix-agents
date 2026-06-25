"""Lightweight profiling primitives for CrafText/JAX module phases.

The project uses this layer for local evidence and notebook introspection.  It
is intentionally small: wall-clock phase timings always work on CPU, while NVTX
annotations are enabled opportunistically inside NVIDIA/JAX-Toolbox containers.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax


@dataclass(frozen=True)
class ProfileEvent:
    """Aggregated timing evidence for one named phase."""

    name: str
    calls: int
    total_seconds: float
    max_seconds: float

    @property
    def mean_seconds(self) -> float:
        """Mean wall time per call."""
        return self.total_seconds / self.calls if self.calls else 0.0

    def as_dict(self) -> dict[str, float | int | str]:
        """Return a stable JSON-serializable representation."""
        return {
            "name": self.name,
            "calls": self.calls,
            "total_seconds": self.total_seconds,
            "mean_seconds": self.mean_seconds,
            "max_seconds": self.max_seconds,
        }


class PhaseProfiler:
    """Hierarchical wall-clock profiler with optional NVTX ranges.

    :param enable_nvtx: If true, emit NVTX ranges when the optional ``nvtx``
        package is installed. Missing ``nvtx`` never breaks CPU/unit paths.
    """

    def __init__(self, *, enable_nvtx: bool = False) -> None:
        self._stack: list[str] = []
        self._totals: defaultdict[str, float] = defaultdict(float)
        self._maxima: defaultdict[str, float] = defaultdict(float)
        self._calls: defaultdict[str, int] = defaultdict(int)
        self._nvtx = _load_nvtx() if enable_nvtx else None

    @contextmanager
    def section(self, name: str, *, sync_result: object | None = None) -> Iterator[None]:
        """Record one named phase.

        :param name: Segment name. Nested sections are emitted as dot paths.
        :param sync_result: Optional JAX PyTree to ``block_until_ready`` before
            stopping the timer; use this when timing asynchronous device work.
        """
        if not name or "." in name:
            raise ValueError("section name must be non-empty and cannot contain '.'")
        self._stack.append(name)
        path = ".".join(self._stack)
        start = time.perf_counter()
        with _nvtx_range(self._nvtx, path):
            try:
                yield
                if sync_result is not None:
                    block_until_ready(sync_result)
            finally:
                elapsed = time.perf_counter() - start
                self._totals[path] += elapsed
                self._maxima[path] = max(self._maxima[path], elapsed)
                self._calls[path] += 1
                self._stack.pop()

    def events(self) -> tuple[ProfileEvent, ...]:
        """Return timing events sorted by phase path."""
        return tuple(
            ProfileEvent(
                name=name,
                calls=self._calls[name],
                total_seconds=self._totals[name],
                max_seconds=self._maxima[name],
            )
            for name in sorted(self._totals)
        )

    def summary(self) -> dict[str, dict[str, float | int | str]]:
        """Return a path-keyed timing summary suitable for notebooks."""
        return {event.name: event.as_dict() for event in self.events()}

    def reset(self) -> None:
        """Clear all collected timings."""
        self._stack.clear()
        self._totals.clear()
        self._maxima.clear()
        self._calls.clear()


def block_until_ready(value: object) -> object:
    """Synchronize a JAX value or PyTree and return it unchanged."""
    return jax.tree.map(_block_leaf, value)


def save_profile(
    path: Path, events: tuple[ProfileEvent, ...], *, metadata: Mapping[str, Any]
) -> None:
    """Persist profiling evidence as stable JSON."""
    payload = {
        "schema": "tunix-craftext.profile/v1",
        "metadata": dict(metadata),
        "events": [event.as_dict() for event in events],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _block_leaf(value: object) -> object:
    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    return value


def _load_nvtx() -> object | None:
    try:
        import nvtx  # type: ignore[import-not-found]
    except ImportError:
        return None
    return nvtx


@contextmanager
def _nvtx_range(nvtx_module: object | None, name: str) -> Iterator[None]:
    if nvtx_module is None:
        with nullcontext():
            yield
        return
    annotate = getattr(nvtx_module, "annotate")
    with annotate(name):
        yield
