"""Versioned JSON replay artifacts for prompt-driven environment trajectories.

This module persistently records prompt/model/action trajectories with schema
metadata and provenance so experiments can be replayed, inspected, and debugged
after the fact.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ReplayStep:
    """One prompt/model/environment decision retained for deterministic inspection.

    :ivar index: int
    :ivar prompt: str
    :ivar raw_completion: str
    :ivar action_id: int
    :ivar action_label: str
    :ivar reward: float
    :ivar terminated: bool
    :ivar truncated: bool
    :ivar action_mask: tuple[bool, ...] | None
    :ivar observation: object | None
    :ivar invalid_format: int
    :ivar unknown_action: int
    :ivar masked_action: int
    :ivar fallback_used: bool

    Example:
        >>> obj = ReplayStep(index=0, prompt="Do X", raw_completion="<action>...")
    """

    index: int
    prompt: str
    raw_completion: str
    action_id: int
    action_label: str
    reward: float
    terminated: bool
    truncated: bool = False
    action_mask: tuple[bool, ...] | None = None
    observation: object | None = None
    invalid_format: int = 0
    unknown_action: int = 0
    masked_action: int = 0
    fallback_used: bool = False
    token_logprobs: tuple[float, ...] | None = None
    token_ids: tuple[int, ...] | None = None
    prompt_token_ids: tuple[int, ...] | None = None


@dataclass(frozen=True)
class ReplayArtifact:
    """Versioned replay with code/config provenance and ordered decision steps.

    :ivar config_path: str
    :ivar commit: str
    :ivar backend: str
    :ivar steps: tuple[ReplayStep, ...]
    :ivar schema: str

    Example:
        >>> obj = ReplayArtifact(
        ...     config_path="configs/exp.yaml",
        ...     commit="abcd123",
        ...     backend="scripted",
        ...     steps=(ReplayStep(...),),
        ... )
    """

    config_path: str
    commit: str
    backend: str
    steps: tuple[ReplayStep, ...]
    schema: str = "tunix-craftext.replay/v3"


def _serialize_value(value: object) -> object:
    if isinstance(value, dict):
        return {key: _serialize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    try:
        array = np.asarray(value)
    except Exception:
        return value
    if isinstance(array, np.ndarray):
        return array.tolist()
    return value


def save_replay(path: Path, artifact: ReplayArtifact) -> None:
    """Persist a human-inspectable replay as atomic JSON.

    The function writes a temporary file next to `path` and then renames it to
    ensure the operation is atomic on POSIX filesystems.

    :param path: Destination file path for the replay JSON.
    :param artifact: `ReplayArtifact` instance to serialize.
    :returns: None
    :raises OSError: If the filesystem write or rename fails.

    Example:
        >>> save_replay(Path("out/replay.json"), artifact)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    payload = _serialize_value(asdict(artifact))
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)
