"""Process-level memory initialization for one-process JAX/vLLM runs.

The project often runs JAX, Tunix and vLLM inside one Python process during
notebook/debug work.  GPU memory policy must be configured before any of those
frameworks initialize their CUDA runtime; this module therefore has no JAX,
Torch or vLLM imports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

MemoryEnvironmentKey = Literal[
    "XLA_PYTHON_CLIENT_PREALLOCATE",
    "XLA_PYTHON_CLIENT_MEM_FRACTION",
    "VLLM_WORKER_MULTIPROC_METHOD",
]


@dataclass(frozen=True)
class MonolithMemoryConfig:
    """Effective process environment for shared JAX/vLLM GPU memory.

    :param jax_fraction: Fraction of GPU memory JAX may reserve when XLA needs
        a pool.  ``XLA_PYTHON_CLIENT_PREALLOCATE=false`` still prevents eager
        reservation at startup.
    :param preallocate: XLA preallocation flag, normally ``False`` for vLLM
        side-by-side rollout.
    :param vllm_worker_multiproc_method: vLLM worker start method. ``spawn`` is
        safest after JAX has initialized threads.
    """

    jax_fraction: float = 0.3
    preallocate: bool = False
    vllm_worker_multiproc_method: str = "spawn"

    def __post_init__(self) -> None:
        """Validate configuration before mutating process environment."""
        if not 0.0 < self.jax_fraction <= 1.0:
            raise ValueError("jax_fraction must be in (0, 1]")
        if self.vllm_worker_multiproc_method not in {"spawn", "forkserver", "fork"}:
            raise ValueError(
                "vllm_worker_multiproc_method must be 'spawn', 'forkserver' or 'fork'"
            )


def setup_monolith_memory(
    *,
    jax_fraction: float = 0.3,
    preallocate: bool = False,
    vllm_worker_multiproc_method: str = "spawn",
    force: bool = False,
) -> MonolithMemoryConfig:
    """Configure JAX/vLLM memory env vars before framework initialization.

    Call this as the first executable line in scripts/notebooks that run JAX and
    vLLM in one process.  Existing user-supplied environment variables are
    preserved unless ``force=True``.

    :returns: The validated effective config requested by the caller.
    """
    config = MonolithMemoryConfig(
        jax_fraction=jax_fraction,
        preallocate=preallocate,
        vllm_worker_multiproc_method=vllm_worker_multiproc_method,
    )
    _set_env(
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "true" if config.preallocate else "false",
        force=force,
    )
    _set_env("XLA_PYTHON_CLIENT_MEM_FRACTION", str(config.jax_fraction), force=force)
    _set_env(
        "VLLM_WORKER_MULTIPROC_METHOD",
        config.vllm_worker_multiproc_method,
        force=force,
    )
    return config


def monolith_memory_environment() -> dict[MemoryEnvironmentKey, str | None]:
    """Return the current relevant memory/process environment values."""
    keys: tuple[MemoryEnvironmentKey, ...] = (
        "XLA_PYTHON_CLIENT_PREALLOCATE",
        "XLA_PYTHON_CLIENT_MEM_FRACTION",
        "VLLM_WORKER_MULTIPROC_METHOD",
    )
    return {key: os.environ.get(key) for key in keys}


def _set_env(key: MemoryEnvironmentKey, value: str, *, force: bool) -> None:
    """Set one env var while respecting explicit user configuration."""
    if force:
        os.environ[key] = value
    else:
        os.environ.setdefault(key, value)
