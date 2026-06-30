#!/usr/bin/env python3
"""Inspect target-platform libraries for vLLM/Tunix accelerator runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from tunix_craftext.diagnostics.accelerator_stack import (
    DEFAULT_EXTRAS,
    DEFAULT_PROBES,
    build_accelerator_stack_report,
)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse stack-inspection arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing pyproject.toml.",
    )
    parser.add_argument(
        "--extra",
        action="append",
        default=None,
        help="Optional-dependency extra to include. Defaults to target vLLM stack extras.",
    )
    parser.add_argument(
        "--probe",
        action="append",
        default=None,
        help="Import name to probe. Defaults to JAX/Torch/vLLM/CrafText runtime modules.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. The report is always printed to stdout.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero if requirements are missing/unsatisfied or imports are broken.",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Print a JSON accelerator stack report."""
    args = parse_args(arguments)
    report = build_accelerator_stack_report(
        args.project_root,
        extras=tuple(args.extra) if args.extra is not None else DEFAULT_EXTRAS,
        probes=tuple(args.probe) if args.probe is not None else DEFAULT_PROBES,
    )
    text = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 2 if args.strict and not report["summary"]["ok"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
