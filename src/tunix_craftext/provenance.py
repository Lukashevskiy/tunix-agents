"""Expose source revision information to experiments and the documentation build."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


def git_revision(root: Optional[Path] = None) -> str:
    """Return the checked-out revision, or a clear sentinel outside a Git checkout."""
    root = root or Path.cwd()
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unversioned"
