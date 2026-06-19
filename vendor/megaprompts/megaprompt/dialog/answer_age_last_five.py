from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def _format_steps_ago(turn: dict[str, Any]) -> str:
    steps_ago = turn.get("answer_steps_ago")
    if isinstance(steps_ago, int) and steps_ago >= 0:
        if steps_ago == 1:
            return " (answer received 1 step ago)"
        return f" (answer received {steps_ago} steps ago)"
    return ""


def render(dialog: Any) -> str:
    if dialog is None:
        return ""
    if isinstance(dialog, str):
        return dialog.strip()
    if isinstance(dialog, dict):
        history = [dialog]
    elif isinstance(dialog, Iterable):
        history = [turn for turn in dialog if isinstance(turn, dict)]
    else:
        return str(dialog).strip()

    lines: list[str] = []
    for turn in history[-5:]:
        agent_msg = str(
            turn.get("agent")
            or turn.get("question")
            or (turn.get("content") if str(turn.get("role", "")).lower() in {"user", "agent"} else "")
            or ""
        ).strip()
        oracle_msg = str(
            turn.get("oracle")
            or turn.get("answer")
            or (
                turn.get("content")
                if str(turn.get("role", "")).lower() in {"assistant", "oracle", "operator"}
                else ""
            )
            or ""
        ).strip()
        if agent_msg:
            lines.append(f"Agent: {agent_msg}")
        if oracle_msg:
            lines.append(f"Operator: {oracle_msg}{_format_steps_ago(turn)}")
    return "\n".join(lines).strip()
