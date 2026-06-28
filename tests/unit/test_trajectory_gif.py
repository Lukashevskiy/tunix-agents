"""Trajectory GIF export contracts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tunix_craftext.artifacts.trajectory_gif import (
    frames_from_replay_payload,
    load_replay_payload,
    normalize_observation_image,
    scale_frame,
    write_gif,
)


def test_normalize_observation_image_converts_grayscale_to_rgb() -> None:
    frame = normalize_observation_image([[0.0, 1.0], [2.0, 3.0]])

    assert frame is not None
    assert frame.shape == (2, 2, 3)
    assert frame.dtype.name == "uint8"
    assert frame[0, 0].tolist() == [0, 0, 0]
    assert frame[-1, -1].tolist() == [255, 255, 255]


def test_frames_from_replay_payload_skips_text_only_steps_and_scales() -> None:
    payload = {
        "steps": [
            {"observation": "not an image"},
            {"observation": [[[0, 0, 0], [255, 0, 0]]]},
        ]
    }

    [frame] = frames_from_replay_payload(payload, scale=2)

    assert frame.shape == (2, 4, 3)
    assert frame[0, 0].tolist() == [0, 0, 0]
    assert frame[0, -1].tolist() == [255, 0, 0]


def test_gif_writer_persists_animation(tmp_path: Path) -> None:
    payload = {
        "steps": [
            {"observation": [[0, 1], [2, 3]]},
            {"observation": [[3, 2], [1, 0]]},
        ]
    }
    frames = frames_from_replay_payload(payload, scale=1)
    output = tmp_path / "trajectory.gif"

    write_gif(output, frames, fps=5)

    assert output.is_file()
    assert output.read_bytes().startswith(b"GIF")


def test_load_replay_payload_validates_steps(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"steps": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid steps"):
        load_replay_payload(path)


def test_scale_and_writer_reject_invalid_arguments(tmp_path: Path) -> None:
    frame = normalize_observation_image([[1]])
    assert frame is not None
    with pytest.raises(ValueError, match="scale"):
        scale_frame(frame, 0)
    with pytest.raises(ValueError, match="without renderable frames"):
        write_gif(tmp_path / "empty.gif", ())
