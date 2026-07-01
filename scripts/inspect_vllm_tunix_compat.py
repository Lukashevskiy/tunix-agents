#!/usr/bin/env python3
"""Inspect whether the installed vLLM build matches Tunix GRPO weight-sync needs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from tunix_craftext.diagnostics import build_vllm_tunix_compat_report


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse vLLM/Tunix compatibility inspection arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. The report is always printed to stdout.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when Tunix vLLM weight-sync hooks are not available.",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Print one JSON vLLM/Tunix compatibility report."""
    args = parse_args(arguments)
    report = build_vllm_tunix_compat_report()
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 2 if args.strict and not report["summary"]["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
