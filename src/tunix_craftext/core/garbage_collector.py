"""Best-effort memory cleanup between rollout and train phases."""

from __future__ import annotations

import gc
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuCacheCleanupReport:
    """Summary of optional GPU cache cleanup work.

    :param gc_collected: Number of Python objects collected by ``gc.collect``.
    :param torch_available: Whether importing torch succeeded.
    :param cuda_available: Whether torch reported CUDA availability.
    :param torch_cuda_cache_cleared: Whether ``empty_cache`` was invoked.
    :param torch_ipc_collected: Whether ``ipc_collect`` was invoked.
    """

    gc_collected: int
    torch_available: bool
    cuda_available: bool
    torch_cuda_cache_cleared: bool
    torch_ipc_collected: bool


def clean_gpu_cache() -> GpuCacheCleanupReport:
    """Collect Python garbage and clear Torch CUDA cache when available.

    The function is intentionally optional-dependency safe: CPU/macOS tests can
    call it without installing torch, while target GPU runners get explicit
    ``torch.cuda.empty_cache``/``ipc_collect`` cleanup between vLLM rollout and
    Tunix/JAX train phases.
    """
    collected = gc.collect()
    try:
        import torch  # type: ignore[import-not-found]
    except Exception:
        return GpuCacheCleanupReport(
            gc_collected=collected,
            torch_available=False,
            cuda_available=False,
            torch_cuda_cache_cleared=False,
            torch_ipc_collected=False,
        )

    cuda_available = bool(torch.cuda.is_available())
    cache_cleared = False
    ipc_collected = False
    if cuda_available:
        torch.cuda.empty_cache()
        cache_cleared = True
        ipc_collect = getattr(torch.cuda, "ipc_collect", None)
        if callable(ipc_collect):
            ipc_collect()
            ipc_collected = True
    return GpuCacheCleanupReport(
        gc_collected=collected,
        torch_available=True,
        cuda_available=cuda_available,
        torch_cuda_cache_cleared=cache_cleared,
        torch_ipc_collected=ipc_collected,
    )
