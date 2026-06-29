#!/usr/bin/env python3
"""Check target-server readiness before expensive Tunix CrafText runs."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from tunix_craftext.training.server_readiness import check_server_readiness


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse readiness-check options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        type=Path,
        default=Path("configs/grpo/qwen_agentic_local.yaml"),
        help="Canonical Agentic GRPO profile to validate.",
    )
    parser.add_argument(
        "--mode",
        choices=("evidence", "scripted"),
        default="evidence",
        help="evidence writes logging probes only; scripted also runs CrafText tool-loop smoke.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Override evidence directory instead of profile.evidence.root.",
    )
    parser.add_argument(
        "--require-accelerator",
        action="store_true",
        help="Fail if JAX resolves to CPU. Use this on the Pro 6000 server.",
    )
    parser.add_argument(
        "--require-snapshot",
        action="store_true",
        help="Fail if profile.model.snapshot is missing.",
    )
    parser.add_argument(
        "--scripted-horizon",
        type=int,
        default=2,
        help="Short horizon for --mode scripted validation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON report path; stdout always prints the same report.",
    )
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run readiness checks and return a shell-friendly exit code."""
    args = parse_args(arguments)
    report = check_server_readiness(
        args.profile,
        mode=args.mode,
        run_dir=args.run_dir,
        require_accelerator=args.require_accelerator,
        require_snapshot=args.require_snapshot,
        scripted_horizon=args.scripted_horizon,
    )
    payload = report.to_json_dict()
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    sys.stdout.write(text)
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
