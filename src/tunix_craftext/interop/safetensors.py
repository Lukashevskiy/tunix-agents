"""Optional safe loader for HuggingFace-style safetensors checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .template import ConversionError


def load_safetensors(path: Path) -> dict[str, Any]:
    """Load a safetensors file without supporting pickle-based checkpoint formats."""
    try:
        from safetensors.numpy import load_file
    except ImportError as error:
        raise ConversionError("install `tunix-craftext[interop]` to load safetensors files") from error
    return dict(load_file(str(path)))
