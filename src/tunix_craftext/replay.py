"""Versioned JSON replay artifacts for prompt-driven environment trajectories."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReplayStep:
    """One prompt/model/environment decision retained for deterministic inspection."""

    index: int
    prompt: str
    raw_completion: str
    action_id: int
    action_label: str
    reward: float
    terminated: bool
    invalid_format: int = 0
    unknown_action: int = 0
    fallback_used: bool = False
    token_logprobs: tuple[float, ...] | None = None
    token_ids: tuple[int, ...] | None = None
    prompt_token_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ReplayArtifact:
    """Versioned replay with code/config provenance and ordered decision steps."""

    config_path: str
    commit: str
    backend: str
    steps: tuple[ReplayStep, ...]
    schema: str = "tunix-craftext.replay/v3"


def save_replay(path: Path, artifact: ReplayArtifact) -> None:
    """Persist a human-inspectable replay atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(asdict(artifact), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)
