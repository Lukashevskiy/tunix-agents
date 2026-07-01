"""External-rollout GRPO evidence built from replay artifacts.

This module is the safe bridge between the working standalone inference lane
(``VllmInferenceEngine``/``BatchLlmBackend``) and the trainer-facing GRPO/PPO
stack.  It intentionally does not instantiate Tunix ``RLCluster``: direct vLLM
rollout can collect trajectories today, while Tunix weight-sync compatibility
is validated separately.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from ..artifacts.replay import ReplayArtifact
from ..rollouts.batched import BatchedTextRollout, replays_from_batched_rollout


class ExternalGrpoError(ValueError):
    """Raised when replay evidence cannot form a valid external GRPO batch."""


@dataclass(frozen=True)
class ExternalGrpoSample:
    """One sampled completion/trajectory for a fixed GRPO task.

    :ivar group_id: Stable task group identifier.
    :ivar sample_id: Generation index inside the group.
    :ivar total_reward: Sum of environment rewards over the replay.
    :ivar advantage: Group-normalized GRPO advantage.
    :ivar replay: Full prompt/completion/action/environment evidence.
    """

    group_id: str
    sample_id: int
    total_reward: float
    advantage: float
    replay: ReplayArtifact


@dataclass(frozen=True)
class ExternalGrpoGroup:
    """A GRPO task group containing multiple independently sampled trajectories.

    :ivar group_id: Stable task group identifier.
    :ivar goal: Text goal/instruction shown to the policy.
    :ivar samples: At least two sampled trajectories for the same task.
    :ivar mean_reward: Mean total reward across samples.
    :ivar std_reward: Population reward standard deviation before epsilon.
    """

    group_id: str
    goal: str
    samples: tuple[ExternalGrpoSample, ...]
    mean_reward: float
    std_reward: float

    def __post_init__(self) -> None:
        """Validate group cardinality and advantage alignment."""
        if len(self.samples) < 2:
            raise ExternalGrpoError("GRPO groups require at least two samples")
        if any(sample.group_id != self.group_id for sample in self.samples):
            raise ExternalGrpoError("all samples must use the group id")


@dataclass(frozen=True)
class ExternalGrpoBatch:
    """Ordered external-rollout GRPO evidence for one learning/evaluation slice."""

    groups: tuple[ExternalGrpoGroup, ...]
    schema: str = "tunix-craftext.external-grpo/v1"

    def __post_init__(self) -> None:
        """Reject empty batches before downstream training code sees them."""
        if not self.groups:
            raise ExternalGrpoError("external GRPO batch must contain at least one group")

    @property
    def sample_count(self) -> int:
        """Return total number of replay samples in the batch."""
        return sum(len(group.samples) for group in self.groups)

    @property
    def mean_reward(self) -> float:
        """Return mean reward over all samples."""
        rewards = [sample.total_reward for group in self.groups for sample in group.samples]
        return sum(rewards) / len(rewards)


def group_normalized_advantages(
    rewards: Sequence[float], *, epsilon: float = 1e-6
) -> tuple[float, ...]:
    """Compute GRPO-style normalized advantages for one task group.

    Identical rewards intentionally produce all-zero advantages: there is no
    within-task preference signal to push the actor.

    :param rewards: Total reward per sampled trajectory in one group.
    :param epsilon: Numerical guard added to a non-zero standard deviation.
    :returns: Advantage for each reward, preserving input order.
    :raises ExternalGrpoError: If fewer than two generations are supplied.
    """
    if len(rewards) < 2:
        raise ExternalGrpoError("GRPO grouped advantages require at least two rewards")
    mean = sum(float(reward) for reward in rewards) / len(rewards)
    variance = sum((float(reward) - mean) ** 2 for reward in rewards) / len(rewards)
    std = math.sqrt(variance)
    if std < epsilon:
        return tuple(0.0 for _ in rewards)
    return tuple((float(reward) - mean) / (std + epsilon) for reward in rewards)


def external_grpo_group_from_replays(
    *,
    goal: str,
    group_id: str,
    replays: Sequence[ReplayArtifact],
    require_token_provenance: bool = True,
) -> ExternalGrpoGroup:
    """Build one GRPO task group from multiple replay trajectories.

    :param goal: Task/instruction text shared by the group.
    :param group_id: Stable group identifier written to evidence.
    :param replays: Replay artifacts sampled for the same task.
    :param require_token_provenance: When true, every replay step must include
        generated token ids, prompt token ids and token log-probabilities.
    :returns: ExternalGrpoGroup with normalized advantages.
    :raises ExternalGrpoError: If cardinality or token evidence is invalid.
    """
    if len(replays) < 2:
        raise ExternalGrpoError("external GRPO group needs at least two replay samples")
    totals = tuple(_total_reward(replay) for replay in replays)
    advantages = group_normalized_advantages(totals)
    if require_token_provenance:
        for sample_id, replay in enumerate(replays):
            _validate_token_provenance(replay, sample_id=sample_id)
    samples = tuple(
        ExternalGrpoSample(
            group_id=group_id,
            sample_id=sample_id,
            total_reward=total_reward,
            advantage=advantages[sample_id],
            replay=replay,
        )
        for sample_id, (replay, total_reward) in enumerate(
            zip(replays, totals, strict=True)
        )
    )
    mean = sum(totals) / len(totals)
    variance = sum((reward - mean) ** 2 for reward in totals) / len(totals)
    return ExternalGrpoGroup(
        group_id=group_id,
        goal=goal,
        samples=samples,
        mean_reward=mean,
        std_reward=math.sqrt(variance),
    )


def external_grpo_batch_from_replays(
    *,
    goal: str,
    group_prefix: str,
    replays: Sequence[ReplayArtifact],
    group_size: int,
    require_token_provenance: bool = True,
) -> ExternalGrpoBatch:
    """Chunk ordered replays into GRPO groups.

    :param goal: Task/instruction text shared by produced groups.
    :param group_prefix: Prefix used to create stable group ids.
    :param replays: Ordered replay artifacts.
    :param group_size: Number of samples per GRPO group; must be at least two
        and divide ``len(replays)`` exactly.
    :param require_token_provenance: Forwarded to
        :func:`external_grpo_group_from_replays`.
    :returns: ExternalGrpoBatch containing one or more groups.
    """
    if group_size < 2:
        raise ExternalGrpoError("group_size must be at least two")
    if not replays or len(replays) % group_size != 0:
        raise ExternalGrpoError("replay count must be a positive multiple of group_size")
    groups = []
    for group_index, start in enumerate(range(0, len(replays), group_size)):
        groups.append(
            external_grpo_group_from_replays(
                goal=goal,
                group_id=f"{group_prefix}-{group_index}",
                replays=tuple(replays[start : start + group_size]),
                require_token_provenance=require_token_provenance,
            )
        )
    return ExternalGrpoBatch(tuple(groups))


def external_grpo_batch_from_batched_rollout(
    rollout: BatchedTextRollout,
    *,
    goal: str,
    group_prefix: str,
    group_size: int,
    config_path: str,
    commit: str,
    backend: str,
    require_token_provenance: bool = True,
) -> ExternalGrpoBatch:
    """Convert a batched text rollout directly into grouped GRPO evidence."""
    replays = replays_from_batched_rollout(
        rollout,
        config_path=config_path,
        commit=commit,
        backend=backend,
    )
    return external_grpo_batch_from_replays(
        goal=goal,
        group_prefix=group_prefix,
        replays=replays,
        group_size=group_size,
        require_token_provenance=require_token_provenance,
    )


def summarize_external_grpo_batch(batch: ExternalGrpoBatch) -> dict[str, object]:
    """Return compact JSON-safe metrics for logging and dashboards."""
    advantages = [
        sample.advantage for group in batch.groups for sample in group.samples
    ]
    rewards = [
        sample.total_reward for group in batch.groups for sample in group.samples
    ]
    return {
        "schema": batch.schema,
        "group_count": len(batch.groups),
        "sample_count": batch.sample_count,
        "mean_reward": batch.mean_reward,
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "mean_abs_advantage": sum(abs(value) for value in advantages) / len(advantages),
    }


def save_external_grpo_batch(path: Path, batch: ExternalGrpoBatch) -> None:
    """Persist external GRPO evidence as stable human-readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(asdict(batch), indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _total_reward(replay: ReplayArtifact) -> float:
    if not replay.steps:
        raise ExternalGrpoError("replay samples must contain at least one step")
    return sum(float(step.reward) for step in replay.steps)


def _validate_token_provenance(replay: ReplayArtifact, *, sample_id: int) -> None:
    for step_index, step in enumerate(replay.steps):
        if not step.token_ids:
            raise ExternalGrpoError(
                f"sample {sample_id} step {step_index} lacks generated token ids"
            )
        if not step.prompt_token_ids:
            raise ExternalGrpoError(
                f"sample {sample_id} step {step_index} lacks prompt token ids"
            )
        if step.token_logprobs is None:
            raise ExternalGrpoError(
                f"sample {sample_id} step {step_index} lacks token logprobs"
            )
        if len(step.token_ids) != len(step.token_logprobs):
            raise ExternalGrpoError(
                f"sample {sample_id} step {step_index} token/logprob lengths differ"
            )
