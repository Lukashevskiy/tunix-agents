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

Build the local documentation site with `mkdocs serve` (after `pip install -e '.[docs]'`).
Use `make docs` instead for the full dashboard build: it refreshes Git status, plan progress,
capability inventory and benchmark tables before running MkDocs. Benchmark JSON artifacts placed
under `artifacts/benchmarks/` appear automatically on the next build. The GitHub Pages workflow
does the same on each push to `main`, weekly, or via manual dispatch.

Every change follows the repository [Definition of Done](docs/delivery.md): audit, applicable
tests and performance evidence, documentation/status updates, intentional commit and site build.

Read [the execution plan](docs/plan.md), [architecture](docs/architecture.md), and
[test/benchmark strategy](docs/quality.md) before extending the trainer.
