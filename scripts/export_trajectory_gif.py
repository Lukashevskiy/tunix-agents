#!/usr/bin/env python3
"""Export a saved replay trajectory observation stream as an animated GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

from tunix_craftext.trajectory_gif import (
    frames_from_replay_payload,
    load_replay_payload,
    write_gif,
)


def parse_args() -> argparse.Namespace:
    """Parse GIF export CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("artifacts/trajectories/manual-craftext-latest.json"),
        help="Path to a saved replay JSON trajectory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/visualizations/trajectory-latest.gif"),
        help="Destination GIF path.",
    )
    parser.add_argument("--fps", type=float, default=4.0, help="GIF frames per second.")
    parser.add_argument(
        "--scale",
        type=int,
        default=4,
        help="Integer nearest-neighbour scale for tiny CrafText observations.",
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="GIF loop count; 0 means loop forever.",
    )
    return parser.parse_args()


def main() -> None:
    """Load replay observations and write an animated GIF artifact."""
    args = parse_args()
    payload = load_replay_payload(args.trajectory)
    frames = frames_from_replay_payload(payload, scale=args.scale)
    write_gif(args.output, frames, fps=args.fps, loop=args.loop)
    print(f"gif: {args.output}")
    print(f"frames: {len(frames)}")


if __name__ == "__main__":
    main()
