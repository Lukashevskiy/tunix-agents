"""Expose source revision information to experiments and the documentation build.

This module reads the current Git checkout revision and returns a clear sentinel
when the code is not running inside a repository, making run provenance explicit
and reproducible.
"""

from __future__ import annotations

import subprocess
from pathlib import Path


def git_revision(root: Path | None = None) -> str:
    """Return the checked-out revision, or a clear sentinel outside a Git checkout.

    :param root: Path | None input value
    :returns: str

    Example:
    >>> result = git_revision(root)"""
    root = root or Path.cwd()
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unversioned"
