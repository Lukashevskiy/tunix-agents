"""Optional safe loader for HuggingFace-style safetensors checkpoints.

This module provides a minimal wrapper around safetensors loading to avoid
pickle-based formats while still producing arrays that can be normalized into
JAX-friendly tensors.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np

from .template import ConversionError


def load_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Load a safetensors file without supporting pickle-based checkpoint formats.

    :param path: Path input value
    :returns: dict[str, np.ndarray]

    Example:
        >>> result = load_safetensors(path)"""
    try:
        from safetensors.numpy import load_file  # type: ignore[import-not-found]
    except ImportError as error:
        raise ConversionError(
            "install `tunix-craftext[interop]` to load safetensors files"
        ) from error
    return cast(dict[str, np.ndarray], dict(load_file(str(path))))
