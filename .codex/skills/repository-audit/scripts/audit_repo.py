#!/usr/bin/env python3
"""Read-only structural audit for Tunix CrafText."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path.cwd()
REQUIRED = ("pyproject.toml", "docs/plan.md", "docs/project_status.json", "mkdocs.yml")


def git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main() -> None:
    findings = []
    for relative in REQUIRED:
        if not (ROOT / relative).is_file():
            findings.append({"severity": "error", "check": "required-file", "detail": f"missing {relative}"})
    if not git("rev-parse", "--is-inside-work-tree"):
        findings.append({"severity": "error", "check": "git", "detail": "not inside a Git worktree"})
    dirty = git("status", "--porcelain")
    if dirty:
        findings.append({"severity": "warning", "check": "git-clean", "detail": "working tree has uncommitted changes"})
    plan = (ROOT / "docs/plan.md").read_text(encoding="utf-8") if (ROOT / "docs/plan.md").exists() else ""
    if "- [" not in plan:
        findings.append({"severity": "warning", "check": "roadmap", "detail": "no checkbox tasks found"})
    status_path = ROOT / "docs/project_status.json"
    if status_path.exists():
        try:
            ready = [item for item in json.loads(status_path.read_text())["capabilities"] if item["status"] == "ready"]
            if not ready:
                findings.append({"severity": "warning", "check": "capabilities", "detail": "no ready capabilities declared"})
        except (KeyError, TypeError, json.JSONDecodeError):
            findings.append({"severity": "error", "check": "capabilities", "detail": "invalid project_status.json"})
    print(json.dumps({"repository": str(ROOT), "revision": git("rev-parse", "--short", "HEAD") or "unversioned", "findings": findings}, indent=2))


if __name__ == "__main__":
    main()
