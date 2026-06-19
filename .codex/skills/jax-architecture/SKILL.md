---
name: jax-architecture
description: Design or reshape Tunix CrafText components, algorithms, environment adapters, model interoperability, checkpointing, or performance-critical JAX paths. Use when a request needs an architecture decision, a new module boundary, a design diagram, or a design review in this repository.
---

# JAX architecture

Design a small vertical slice before adding implementation. Read `docs/architecture.md`,
`docs/plan.md`, and the closest existing module before proposing a boundary.

## Workflow

1. State the user-facing outcome, non-goals, input/output PyTrees, ownership and failure modes.
2. Put environment-specific behaviour in `adapters/`, trajectory semantics in `rollout/`,
   framework-free math in `algorithms/`, and external/model-format glue in `interop/`.
   Do not let a lower layer import a higher one.
3. Keep compiled paths pure: no filesystem, logging, mutable global state, dynamic shapes or
   host callbacks inside a future `jax.jit`/`lax.scan` body.
4. Make batch axes explicit. Rollouts are time-major `[T, B, ...]`; document masks, RNG
   splitting, terminal/truncation and bootstrap semantics.
5. Isolate Tunix behind a small adapter and pin its compatibility test. Preserve vendor code;
   use adapters rather than modifying `vendor/`.
6. Produce an ADR when the decision is costly to reverse, changes a public contract, or adds a
   dependency. Update `docs/architecture.md`, `docs/plan.md`, and `docs/project_status.json`
   when the implementation status changes.
7. Treat types and docstrings as the public contract: name shapes/axes/dtypes, PRNG ownership,
   mutation and failures. Use generic types or protocols for extensible PyTrees; do not use `Any`
   beyond a validated external adapter boundary.

## Exit criteria

Before implementation, name the first test, the expected benchmark metric, and the smallest
integration fixture. Do not call a planned capability implemented until its test passes.
