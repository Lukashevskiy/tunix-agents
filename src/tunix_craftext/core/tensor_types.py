"""Shared jaxtyping aliases for fixed project-level tensor contracts.

The aliases name *semantic* axes used by the project rather than implementation
details.  ``time batch`` is the canonical rollout layout, ``batch tokens`` is
the padded text-learning layout, and ``batch actions`` is the legal-action
mask.  Arbitrary vendor/model parameter PyTrees intentionally remain generic.
"""

from typing import TypeAlias

from jax.typing import ArrayLike
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

JaxArrayLike: TypeAlias = ArrayLike
"""External array-like input accepted at adapter/interop boundaries."""

JaxTree: TypeAlias = object
"""Opaque JAX PyTree used when the leaf schema is owned by a vendor or model."""

JaxArray: TypeAlias = Array
"""Unshaped JAX array escape hatch for private helpers and vendor protocols."""

ParameterTree: TypeAlias = "dict[str, JaxArray | ParameterTree]"
"""Nested JAX parameter PyTree with string keys and array leaves."""

SingleActionMask: TypeAlias = Bool[Array, "actions"]
"""Legal-action mask for a single environment state, shape ``[A]``."""

BatchFloat: TypeAlias = Float[Array, "batch"]
"""One floating-point value per environment/minibatch row."""

BatchBool: TypeAlias = Bool[Array, "batch"]
"""One boolean flag per environment/minibatch row."""

BatchInt: TypeAlias = Int[Array, "batch"]
"""One integer id per environment/minibatch row."""

BatchFeatureFloat: TypeAlias = Float[Array, "batch features"]
"""Feature matrix for small smoke actors, shape ``[B, F]``."""

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

TokenBatchLogits: TypeAlias = Float[Array, "batch tokens vocab"]
"""Token-level vocabulary logits ``[B, L, V]``."""

TokenBatchHidden: TypeAlias = Float[Array, "batch tokens hidden"]
"""Token-level hidden states ``[B, L, D]``."""

TokenBatchPositionInt: TypeAlias = Int[Array, "batch tokens"]
"""Absolute token positions for causal LM inputs, shape ``[B, L]``."""

PromptTokenBatchInt: TypeAlias = Int[Array, "batch prompt_tokens"]
"""Padded prompt-token tensor ``[B, P]``."""

PromptTokenBatchBool: TypeAlias = Bool[Array, "batch prompt_tokens"]
"""Padded prompt-token mask ``[B, P]``."""

CausalInputTokenBatchInt: TypeAlias = Int[Array, "batch sequence"]
"""Concatenated prompt+generation token ids ``[B, P + L]``."""

CausalInputPositionBatchInt: TypeAlias = Int[Array, "batch sequence"]
"""Concatenated prompt+generation token positions ``[B, P + L]``."""

CausalLmLogits: TypeAlias = Float[Array, "batch sequence vocab"]
"""Causal LM logits over the full prompt+generation sequence."""

CausalLmHidden: TypeAlias = Float[Array, "batch sequence hidden"]
"""Causal LM hidden states over the full prompt+generation sequence."""

ValueHeadKernel: TypeAlias = Float[Array, "hidden"]
"""Linear critic/value-head kernel ``[D]``."""

ValueHeadBias: TypeAlias = Float[Array, ""]
"""Scalar critic/value-head bias."""

TokenIndexBatchInt: TypeAlias = Int[Array, "batch tokens"]
"""Index tensor aligned with generated tokens, shape ``[B, L]``."""

ActionMask: TypeAlias = Bool[Array, "batch actions"]
"""Legal-action mask with environment and discrete-action axes ``[B, A]``."""

ActionLogits: TypeAlias = Float[Array, "batch actions"]
"""Discrete action logits with environment and action axes ``[B, A]``."""
