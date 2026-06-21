"""Versioned Orbax persistence for learner state and run provenance.

The module saves trainable JAX PyTrees through Orbax and stores the small,
human-readable compatibility record next to them.  A caller supplies the
``TrainState`` template on restore so executable Flax/Optax functions remain
local code rather than being serialized as opaque Python objects.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import orbax.checkpoint as ocp  # type: ignore[import-untyped]
from flax.training.train_state import TrainState

CHECKPOINT_SCHEMA = "tunix-craftext.checkpoint/v1"
_METADATA_NAME = "tunix_craftext_metadata.json"


@dataclass(frozen=True, slots=True)
class CheckpointMetadata:
    """Compatibility record stored with each checkpoint.

    :param run_id: Stable identifier of the producing run.
    :param config_digest: Digest of the validated run configuration.
    :param policy_kind: Public policy implementation name used by the run.
    """

    run_id: str
    config_digest: str
    policy_kind: str
    schema: str = CHECKPOINT_SCHEMA


def save_checkpoint(
    directory: Path,
    state: TrainState,
    metadata: CheckpointMetadata,
) -> Path:
    """Persist trainable state with Orbax and a versioned metadata record.

    :param directory: Destination checkpoint directory.
    :param state: Flax state whose parameters, optimizer state and step are saved.
    :param metadata: Immutable run compatibility record.
    :returns: The written checkpoint directory.
    :raises ValueError: If the supplied metadata has an unsupported schema.
    """
    _validate_metadata(metadata)
    checkpointer = ocp.PyTreeCheckpointer()
    trainable_state = {
        "step": state.step,
        "params": state.params,
        "opt_state": state.opt_state,
    }
    checkpointer.save(directory, trainable_state, force=True)
    (directory / _METADATA_NAME).write_text(
        json.dumps(asdict(metadata), sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return directory


def restore_checkpoint(
    directory: Path,
    state_template: TrainState,
) -> tuple[TrainState, CheckpointMetadata]:
    """Restore a compatible learner state into a local Flax/Optax template.

    :param directory: Existing checkpoint directory.
    :param state_template: Current model state that provides ``apply_fn`` and ``tx``.
    :returns: Restored state and its validated provenance metadata.
    :raises ValueError: If the checkpoint metadata schema is unknown.
    """
    metadata = _load_metadata(directory)
    trainable_template = {
        "step": state_template.step,
        "params": state_template.params,
        "opt_state": state_template.opt_state,
    }
    trainable_state: dict[str, Any] = ocp.PyTreeCheckpointer().restore(
        directory, item=trainable_template
    )
    return (
        state_template.replace(
            step=trainable_state["step"],
            params=trainable_state["params"],
            opt_state=trainable_state["opt_state"],
        ),
        metadata,
    )


def _load_metadata(directory: Path) -> CheckpointMetadata:
    """Read and validate checkpoint metadata from its adjacent JSON file."""
    payload = json.loads((directory / _METADATA_NAME).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("checkpoint metadata must be a JSON object")
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"unsupported checkpoint schema {payload.get('schema')!r}; "
            f"expected {CHECKPOINT_SCHEMA!r}"
        )
    try:
        metadata = CheckpointMetadata(**payload)
    except TypeError as error:
        raise ValueError("invalid checkpoint metadata fields") from error
    _validate_metadata(metadata)
    return metadata


def _validate_metadata(metadata: CheckpointMetadata) -> None:
    """Reject incompatible checkpoint schemas before they reach a learner."""
    if metadata.schema != CHECKPOINT_SCHEMA:
        raise ValueError(
            f"unsupported checkpoint schema {metadata.schema!r}; expected {CHECKPOINT_SCHEMA!r}"
        )
