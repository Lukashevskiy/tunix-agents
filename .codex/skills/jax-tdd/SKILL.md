---
name: jax-tdd
description: Implement or modify Tunix CrafText Python/JAX code with test-driven development. Use when adding algorithms, rollout collection, adapters, model conversion, checkpointing, or fixing a bug where shape, randomness, numerical behaviour, or JIT parity must be proven.
---

# JAX TDD

Turn the desired property into a failing test before changing production code. Read
`docs/quality.md` and follow the repository's test layout.

## Workflow

1. Add a smallest hand-computed unit test in `tests/unit/`; include a negative case for invalid
   shapes, masks or checkpoint metadata where relevant.
2. Implement the pure reference version first. Keep it deterministic and expose PRNG keys
   explicitly instead of relying on global randomness.
3. Add property coverage for PyTree structure, `[T, B, ...]` axes, dtypes, terminal vs
   truncation, and no mutation of supplied parameters.
4. Add JIT/vectorized code only after reference correctness. Compare every leaf with the
   reference using exact equality for discrete values and stated tolerances for floats.
5. Add an integration smoke test only when real CrafText/Tunix/Orbax is involved; mark and skip
   it cleanly when the optional dependency is missing.
6. Add a performance test after correctness. Separate compilation/warmup from steady-state and
   save benchmark JSON under `artifacts/benchmarks/`.
7. Add public annotations and Sphinx-compatible docstrings alongside the test. Use `:param:`,
   `:returns:` and `:raises:` fields, let Sphinx render types from signatures, document array
   shape/axes/dtype, and replace contract `Any` with a generic, protocol or explicit data type.

## Required verification

Run the narrow tests first, then `make verify`. For a hot-path change, also run `make perf` and
save benchmark evidence or explicitly record why profiling is inapplicable. Report skips as
missing environment evidence, not successful integration. Before committing, update the relevant
documentation, roadmap checkbox and capability status, then run `$task-board-sync` / `make
sync-tasks`; only verified behaviour may be `ready`.
