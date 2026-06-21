# Vendor compatibility patches

Vendor snapshots stay source-compatible by default. A local change inside `vendor/` is allowed
only when it fixes a concrete integration blocker, has a regression test inside that vendor, and
is recorded here for eventual upstreaming.

## JAX-vmap world preset compatibility

**Goal:** make the CrafText `tiny_box_oob_no_mobs` preset usable as batched JAX environment state.

1. Add a vendor regression test: two independent resets and one `jax.vmap(environment.step)`.
2. Replace Python scalar conversion/membership in `SolidBlocksBehavior` with JAX array operations.
3. Replace dynamic Python branches in `IntrinsicDynamicsBehavior` with arithmetic that is identity
   for zero deltas.
4. Replace sleep/rest dynamic branches in `InstantRecoveryBehavior` with `jax.lax.cond` returning
   a whole state.
5. Repeat the vendor test after every patch; continue only to the next traceback site.
6. Once it passes, add project integration `B=2 × T=8`, document patch provenance in
   `vendor/manifest.json`, then benchmark and prepare an upstream patch.

Steps 1–4 are complete: the vendor vmap regression and project `B=2 × T=8` reference trajectory
pass. Remaining work is RNG-aware `lax.scan` parity, benchmark evidence and upstream patch export.
