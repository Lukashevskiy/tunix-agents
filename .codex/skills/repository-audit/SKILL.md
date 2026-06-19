---
name: repository-audit
description: Audit the health, reproducibility, documentation integrity, test coverage signals, dependency boundaries, and generated-site readiness of the Tunix CrafText repository. Use for a repository audit, pre-merge review, release readiness check, or when project status may have drifted from code and artifacts.
---

# Repository audit

Run `python .codex/skills/repository-audit/scripts/audit_repo.py` from the repository root
first. It is read-only and emits structured findings.

## Review order

1. Treat `error` findings as blockers. Verify Git cleanliness, mandatory repository files and
   non-empty roadmap before judging code quality.
2. Read the dashboard inputs, not generated pages: `docs/plan.md`, `docs/project_status.json`,
   `artifacts/benchmarks/`, Git history and source tests.
3. Run focused tests and `make docs` with the documented environment. Distinguish a skipped
   optional integration test from a passing integration test.
4. Inspect public claims: every capability marked `ready` must point to implementation and a
   passing test; every benchmark claim must have commit, hardware and config/seed metadata.
5. Inspect dependency boundaries: no training logic in `vendor/`; no optional Tunix/HF/Torch
   import in the core import path; no pickle checkpoint loader.
6. Enforce `docs/delivery.md`: ensure the change has test evidence, applicable benchmark evidence,
   updated documentation/status, an intentional commit and a regenerated site.

## Report format

Lead with blockers, then high/medium findings. For every finding give evidence, impact and a
smallest remediation. Do not modify files during an audit unless the user asks to fix findings.
