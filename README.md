# Tunix CrafText

An intentionally small, JAX-native training foundation for CrafText and CagedCrafText.
It replaces the orchestration layer—not the environments—with a testable trajectory
contract, a pure-JAX rollout path, Optax updates, Orbax checkpoints, and a narrow
adapter seam for Tunix.

## Status

**Foundation / contract-first.** The vendored environments and prompt assets are copied
unchanged under `vendor/`; their licenses and attribution remain there. PPO/GRPO and the
Tunix bridge are planned extension points, not claimed implementations yet.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

For the actual environments and Tunix, install the opt-in extras after pinning a known
compatible accelerator-specific JAX build:

```bash
pip install -e '.[envs,tunix,dev]'
```

## Local project site

Create the local documentation environment once, then use the repository command rather than
running a globally installed MkDocs:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip mkdocs-material sphinx sphinx-autodoc-typehints
make serve
```

`make serve` first regenerates the Dashboard from the current Git commit, roadmap, capability
inventory and `artifacts/benchmarks/*.json`, then opens the local MkDocs server. Use `make docs`
to make the same fully populated static `site/` directory without a server. Benchmark JSON
artifacts placed under `artifacts/benchmarks/` appear automatically on the next build. The GitHub
Pages workflow does the same on each push to `main`, weekly, or via manual dispatch.

Without Make, run `.venv/bin/python scripts/generate_dashboard.py && .venv/bin/python -m mkdocs
serve`. Do not use bare `mkdocs serve`: it can select a global interpreter without Material and
will not refresh the generated dashboard pages.

Every change follows the repository [Definition of Done](docs/delivery.md): audit, applicable
tests and performance evidence, documentation/status updates, intentional commit and site build.

Read [the execution plan](docs/plan.md), [architecture](docs/architecture.md), and
[test/benchmark strategy](docs/quality.md) before extending the trainer.
