"""Shared jaxtyping aliases for fixed project-level tensor contracts.

The aliases name *semantic* axes used by the project rather than implementation
details.  ``time batch`` is the canonical rollout layout, ``batch tokens`` is
the padded text-learning layout, and ``batch actions`` is the legal-action
mask.  Arbitrary vendor/model parameter PyTrees intentionally remain generic.
"""

from typing import TypeAlias

from jaxtyping import Array, Bool, Float, Int, Key, UInt32

ScalarFloat: TypeAlias = Float[Array, ""]
"""A scalar floating-point JAX array."""

ScalarBool: TypeAlias = Bool[Array, ""]
"""A scalar boolean JAX array."""

ScalarInt: TypeAlias = Int[Array, ""]
"""A scalar integer JAX array."""

JaxKey: TypeAlias = Key[Array, ""]
"""One modern JAX PRNG key."""

BatchLegacyKey: TypeAlias = UInt32[Array, "batch 2"]
"""Legacy explicit-key representation used by the current vmapped collectors."""

SingleActionMask: TypeAlias = Bool[Array, "actions"]
"""Legal-action mask for a single environment state, shape ``[A]``."""

BatchFloat: TypeAlias = Float[Array, "batch"]
"""One floating-point value per environment/minibatch row."""

BatchBool: TypeAlias = Bool[Array, "batch"]
"""One boolean flag per environment/minibatch row."""

BatchInt: TypeAlias = Int[Array, "batch"]
"""One integer id per environment/minibatch row."""

TimeBatchFloat: TypeAlias = Float[Array, "time batch"]
"""Time-major floating-point rollout tensor ``[T, B]``."""

TimeBatchBool: TypeAlias = Bool[Array, "time batch"]
"""Time-major boolean rollout tensor ``[T, B]``."""

TokenBatchFloat: TypeAlias = Float[Array, "batch tokens"]
"""Padded token-level floating-point tensor ``[B, L]``."""

TokenBatchBool: TypeAlias = Bool[Array, "batch tokens"]
"""Padded token-level boolean mask ``[B, L]``."""

TokenBatchInt: TypeAlias = Int[Array, "batch tokens"]
"""Padded token-id tensor ``[B, L]``."""

PromptTokenBatchInt: TypeAlias = Int[Array, "batch prompt_tokens"]
"""Padded prompt-token tensor ``[B, P]``."""

PromptTokenBatchBool: TypeAlias = Bool[Array, "batch prompt_tokens"]
"""Padded prompt-token mask ``[B, P]``."""

ActionMask: TypeAlias = Bool[Array, "batch actions"]
"""Legal-action mask with environment and discrete-action axes ``[B, A]``."""
