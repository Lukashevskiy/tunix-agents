#!/usr/bin/env python3
"""Write a compact, reproducible Git provenance record for run and docs artifacts."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def git(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unversioned"


def main() -> None:
    destination = Path(sys.argv[1] if len(sys.argv) > 1 else "artifacts/provenance.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_revision": git(["rev-parse", "HEAD"]),
        "git_dirty": bool(git(["status", "--porcelain"])),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    destination.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
