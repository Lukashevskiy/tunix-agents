"""Optional safe loader for HuggingFace-style safetensors checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np

from .template import ConversionError


def load_safetensors(path: Path) -> dict[str, np.ndarray]:
    """Load a safetensors file without supporting pickle-based checkpoint formats."""
    try:
        from safetensors.numpy import load_file  # type: ignore[import-not-found]
    except ImportError as error:
        raise ConversionError(
            "install `tunix-craftext[interop]` to load safetensors files"
        ) from error
    return cast(dict[str, np.ndarray], dict(load_file(str(path))))
