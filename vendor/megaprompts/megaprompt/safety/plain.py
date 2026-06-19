from __future__ import annotations

from typing import Any


def render(safety: Any) -> str:
    text = "" if safety is None else str(safety).strip()
    if not text:
        return "No additional safety constraints are specified for this episode."
    return text
