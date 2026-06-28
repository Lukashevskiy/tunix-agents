"""Typed JAX replay staging for text trajectories, backed by Flashbax.

The adapter deliberately exposes only the small ``init → add → sample`` surface
needed by the synchronous vanilla pipeline.  It does not make PPO off-policy:
the caller owns the update window and must not mix stale behaviour-policy data
without an explicitly implemented correction rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, cast

import jax

from ..artifacts.text_trajectory import TextTrajectoryBatch


class FlashbaxReplayError(ValueError):
    """Raised when a Flashbax text replay contract is invalid."""


class _ReplaySample(Protocol):
    """Structural type for the public Flashbax sample container."""

    experience: TextTrajectoryBatch


class _ItemBuffer(Protocol):
    """Minimal typed surface provided by ``flashbax.make_item_buffer``."""

    def init(self, example: TextTrajectoryBatch) -> object:
        """Create empty replay state from one unbatched item."""

    def add(self, state: object, batch: TextTrajectoryBatch) -> object:
        """Append a leading batch of independent replay items."""

    def can_sample(self, state: object) -> jax.Array:
        """Report whether the configured minimum item count is available."""

    def sample(self, state: object, key: jax.Array) -> _ReplaySample:
        """Uniformly sample a leading batch of replay items."""


@dataclass(frozen=True)
class FlashbaxTextReplay:
    """A fixed-shape Flashbax item buffer for ``TextTrajectoryBatch`` rows.

    :ivar buffer: Public Flashbax item-buffer implementation behind a narrow protocol.
    :ivar template: Fixed token and prompt widths established at initialization.
    """

    buffer: _ItemBuffer
    template: TextTrajectoryBatch

    def initialize(self) -> object:
        """Allocate empty device replay state from the first template row.

        :returns: Flashbax state suitable for :meth:`add` and :meth:`sample`.
        :raises FlashbaxReplayError: If the template has no leading batch items.
        """
        self.template.validate()
        if self.template.token_ids.shape[0] == 0:
            raise FlashbaxReplayError("template must contain at least one replay item")
        example = jax.tree.map(lambda leaf: leaf[0], self.template)
        return self.buffer.init(example)

    def add(self, state: object, batch: TextTrajectoryBatch) -> object:
        """Append all leading ``batch`` rows as independent replay items.

        :param state: Existing Flashbax replay state.
        :param batch: Fixed-shape text decisions with leading item axis ``[B, ...]``.
        :returns: Updated Flashbax state.
        :raises FlashbaxReplayError: If token/prompt widths differ from the template.
        """
        batch.validate_static()
        expected = self.template.token_ids.shape[1], self.template.prompt_token_ids.shape[1]
        actual = batch.token_ids.shape[1], batch.prompt_token_ids.shape[1]
        if actual != expected:
            raise FlashbaxReplayError(
                "batch token and prompt widths must match the fixed replay template"
            )
        return self.buffer.add(state, batch)

    def can_sample(self, state: object) -> jax.Array:
        """Return a JAX scalar indicating whether the buffer has enough items."""
        return self.buffer.can_sample(state)

    def sample(self, state: object, key: jax.Array) -> TextTrajectoryBatch:
        """Draw a uniformly sampled, typed text mini-batch.

        :param state: Existing Flashbax replay state.
        :param key: JAX PRNG key for uniform sampling.
        :returns: A valid ``TextTrajectoryBatch`` with the configured sample size.
        """
        sampled = self.buffer.sample(state, key).experience
        sampled.validate_static()
        return sampled


def make_text_replay_buffer(
    template: TextTrajectoryBatch,
    *,
    capacity: int,
    min_size: int,
    sample_batch_size: int,
) -> FlashbaxTextReplay:
    """Create a fixed-shape, JIT-compatible Flashbax text item buffer.

    Each leading template row is an independent decision record.  Token and
    prompt widths are static by design; bucket variable-length replays before
    insertion rather than recompiling inside an update.

    :param template: Representative padded text batch defining the item PyTree.
    :param capacity: Maximum number of individual decision items retained.
    :param min_size: Minimum retained items before ``can_sample`` is true.
    :param sample_batch_size: Number of decision items returned by one sample.
    :returns: Typed replay adapter backed by Flashbax.
    :raises FlashbaxReplayError: If configuration is non-positive or Flashbax is unavailable.
    """
    if capacity <= 0 or min_size <= 0 or sample_batch_size <= 0:
        raise FlashbaxReplayError("capacity, min_size and sample_batch_size must be positive")
    if min_size > capacity or sample_batch_size > capacity:
        raise FlashbaxReplayError("min_size and sample_batch_size cannot exceed capacity")
    template.validate()
    try:
        import flashbax  # type: ignore[import-untyped]
    except ImportError as error:
        raise FlashbaxReplayError(
            "Flashbax is optional; install `tunix-craftext[replay]` to use text replay"
        ) from error
    buffer = flashbax.make_item_buffer(
        max_length=capacity,
        min_length=min_size,
        sample_batch_size=sample_batch_size,
        add_batches=True,
    )
    return FlashbaxTextReplay(cast(_ItemBuffer, buffer), template)
