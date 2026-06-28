"""Headless GIF export helpers for replay trajectory observations.

The pygame viewer is convenient for local debugging, but validation evidence and
experiment trackers need a file artifact that can be produced without opening a
window.  This module keeps the conversion deterministic and dependency-light:
JSON replay steps provide observations, observations become RGB ``uint8`` frames,
and Pillow is imported only when a GIF is actually written.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

RgbFrame = NDArray[np.uint8]


def load_replay_payload(path: Path) -> dict[str, Any]:
    """Load a replay JSON artifact and validate the top-level trajectory shape.

    :param path: Path to a saved replay JSON file.
    :returns: Replay payload containing a list-valued ``steps`` field.
    :raises ValueError: If the file is not a replay artifact.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "steps" not in payload:
        raise ValueError(f"Trajectory file {path} is not a replay artifact")
    if not isinstance(payload["steps"], list):
        raise ValueError(f"Trajectory file {path} contains invalid steps")
    return payload


def normalize_observation_image(observation: Any) -> RgbFrame | None:
    """Convert a replay observation to an RGB ``uint8`` frame when possible.

    The function accepts two common forms:

    - ``[H, W]`` scalar observations, normalized to grayscale RGB;
    - ``[H, W, C]`` images with ``C`` in ``{1, 3, 4}``; alpha is dropped.

    Non-image observations return ``None`` so callers can skip text-only steps.
    """
    try:
        array = np.asarray(observation)
    except Exception:
        return None
    if array.size == 0 or array.ndim == 1:
        return None

    if array.ndim == 2:
        image = _normalize_float_image(array)
        return np.stack([image, image, image], axis=-1)
    if array.ndim == 3 and array.shape[2] in {1, 3, 4}:
        image = _normalize_float_image(array)
        if image.shape[2] == 1:
            return np.repeat(image, 3, axis=-1)
        if image.shape[2] == 4:
            return image[:, :, :3]
        return image
    return None


def frames_from_replay_payload(payload: dict[str, Any], *, scale: int = 1) -> tuple[RgbFrame, ...]:
    """Extract renderable observation frames from a replay payload.

    :param payload: Replay JSON payload with ``steps``.
    :param scale: Positive integer nearest-neighbour frame scaling factor.
    :returns: Tuple of RGB frames in trajectory order.
    :raises ValueError: If ``steps`` is invalid or ``scale`` is not positive.
    """
    if scale <= 0:
        raise ValueError("scale must be positive")
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise ValueError("payload must contain list-valued steps")

    frames: list[RgbFrame] = []
    for step in steps:
        if not isinstance(step, dict):
            continue
        frame = normalize_observation_image(step.get("observation"))
        if frame is None:
            continue
        frames.append(scale_frame(frame, scale))
    return tuple(frames)


def scale_frame(frame: RgbFrame, scale: int) -> RgbFrame:
    """Scale an RGB frame by an integer factor using nearest-neighbour repeat."""
    if scale <= 0:
        raise ValueError("scale must be positive")
    if scale == 1:
        return frame
    return np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)


def write_gif(path: Path, frames: tuple[RgbFrame, ...], *, fps: float = 4.0, loop: int = 0) -> None:
    """Write RGB frames to an animated GIF using Pillow.

    :param path: Destination GIF path.
    :param frames: Non-empty tuple of RGB ``uint8`` frames.
    :param fps: Positive frames per second.
    :param loop: GIF loop count, where ``0`` means forever.
    :raises ValueError: If no frames are provided or ``fps`` is invalid.
    :raises RuntimeError: If Pillow is not installed.
    """
    if not frames:
        raise ValueError("cannot write GIF without renderable frames")
    if fps <= 0:
        raise ValueError("fps must be positive")
    try:
        from PIL import Image
    except ImportError as error:  # pragma: no cover - depends on optional extras
        raise RuntimeError(
            "Pillow is required for GIF export; install the examples extra"
        ) from error

    path.parent.mkdir(parents=True, exist_ok=True)
    images = [Image.fromarray(frame, mode="RGB") for frame in frames]
    duration_ms = max(1, int(round(1000.0 / fps)))
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=loop,
    )


def _normalize_float_image(array: NDArray[Any]) -> RgbFrame:
    image = array.astype(np.float32)
    if image.dtype != np.uint8:
        image = image - float(np.min(image))
        maximum = float(np.max(image))
        if maximum > 0:
            image = image / maximum * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)
