#!/usr/bin/env python3
"""Estimate vLLM rollout memory requirements from a generation YAML config."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from tunix_craftext.diagnostics.vllm_memory import estimate_vllm_memory_from_config


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/generation/qwen_vllm_sync.yaml"),
        help="Strict generation YAML config.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA device index to inspect through torch.",
    )
    parser.add_argument(
        "--safety-margin-gib",
        type=float,
        default=1.0,
        help="Free-memory margin kept outside the vLLM reservation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. The estimate is always printed to stdout.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when the reservation does not fit current free memory.",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Print a JSON vLLM memory estimate."""
    args = parse_args(arguments)
    estimate = estimate_vllm_memory_from_config(
        args.config,
        device_index=args.device,
        safety_margin_gib=args.safety_margin_gib,
    )
    payload = estimate.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    if args.strict and estimate.fits_current_free_memory is False:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
