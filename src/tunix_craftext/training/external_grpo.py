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
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp

from ..artifacts.replay import ReplayArtifact
from ..core.tensor_types import BatchFloat, BatchInt, TokenBatchBool, TokenBatchFloat, TokenBatchInt
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


@dataclass(frozen=True)
class ExternalGrpoTokenBatch:
    """Token-level learner input derived from external GRPO replay evidence.

    :ivar token_ids: Padded generated token ids, shape ``[N, T]``.
    :ivar old_logprobs: Behaviour token log-probabilities, shape ``[N, T]``.
    :ivar token_mask: Valid generated-token mask, shape ``[N, T]``.
    :ivar advantages: Group-normalized GRPO advantages broadcast to tokens.
    :ivar prompt_token_ids: Padded prompt token ids, shape ``[N, P]``.
    :ivar prompt_token_mask: Valid prompt-token mask, shape ``[N, P]``.
    :ivar sample_rewards: Total trajectory reward for each row's source sample.
    :ivar group_ids: Integer group index per row, useful for audit metrics.
    :ivar sample_ids: Sample index inside the group per row.
    """

    token_ids: TokenBatchInt
    old_logprobs: TokenBatchFloat
    token_mask: TokenBatchBool
    advantages: TokenBatchFloat
    prompt_token_ids: TokenBatchInt
    prompt_token_mask: TokenBatchBool
    sample_rewards: BatchFloat
    group_ids: BatchInt
    sample_ids: BatchInt

    def validate_static(self) -> None:
        """Validate static axes without reading traced array values."""
        token_shape = tuple(self.token_ids.shape)
        if len(token_shape) != 2 or token_shape[0] == 0 or token_shape[1] == 0:
            raise ExternalGrpoError("token_ids must have non-empty shape [N, T]")
        for name in ("old_logprobs", "token_mask", "advantages"):
            if tuple(getattr(self, name).shape) != token_shape:
                raise ExternalGrpoError(f"{name} must have shape {token_shape}")
        prompt_shape = tuple(self.prompt_token_ids.shape)
        if len(prompt_shape) != 2 or prompt_shape[0] != token_shape[0] or prompt_shape[1] == 0:
            raise ExternalGrpoError("prompt_token_ids must have non-empty shape [N, P]")
        if tuple(self.prompt_token_mask.shape) != prompt_shape:
            raise ExternalGrpoError(f"prompt_token_mask must have shape {prompt_shape}")
        for name in ("sample_rewards", "group_ids", "sample_ids"):
            if tuple(getattr(self, name).shape) != token_shape[:1]:
                raise ExternalGrpoError(f"{name} must have shape {token_shape[:1]}")

    def validate(self) -> None:
        """Validate static axes and semantic mask invariants."""
        self.validate_static()
        if not bool(jnp.any(self.token_mask)):
            raise ExternalGrpoError("token_mask must select at least one generated token")
        if bool(jnp.any(jnp.where(self.token_mask, 0.0, self.advantages) != 0.0)):
            raise ExternalGrpoError("advantages cannot be non-zero on padding tokens")


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


def token_batch_from_external_grpo(batch: ExternalGrpoBatch) -> ExternalGrpoTokenBatch:
    """Convert external GRPO replay evidence into padded token learner rows.

    Each replay step becomes one row. The trajectory-level group-normalized
    advantage is broadcast onto every valid generated token in that row. This
    keeps the algorithm critic-free while preserving token-level actor updates.

    :param batch: External GRPO evidence with token provenance.
    :returns: Padded token batch for :func:`masked_token_grpo_loss`.
    :raises ExternalGrpoError: If replay token evidence is missing or malformed.
    """
    rows: list[tuple[ExternalGrpoSample, tuple[int, ...], tuple[float, ...], tuple[int, ...]]] = []
    group_index_by_id = {group.group_id: index for index, group in enumerate(batch.groups)}
    for group in batch.groups:
        for sample in group.samples:
            _validate_token_provenance(sample.replay, sample_id=sample.sample_id)
            for step in sample.replay.steps:
                assert step.token_ids is not None
                assert step.token_logprobs is not None
                assert step.prompt_token_ids is not None
                rows.append(
                    (
                        sample,
                        step.token_ids,
                        step.token_logprobs,
                        step.prompt_token_ids,
                    )
                )
    if not rows:
        raise ExternalGrpoError("external GRPO batch did not contain replay steps")
    token_width = max(len(tokens) for _, tokens, _, _ in rows)
    prompt_width = max(len(prompt_tokens) for _, _, _, prompt_tokens in rows)
    row_count = len(rows)
    token_ids = jnp.zeros((row_count, token_width), dtype=jnp.int32)
    old_logprobs = jnp.zeros((row_count, token_width), dtype=jnp.float32)
    token_mask = jnp.zeros((row_count, token_width), dtype=bool)
    advantages = jnp.zeros((row_count, token_width), dtype=jnp.float32)
    prompt_token_ids = jnp.zeros((row_count, prompt_width), dtype=jnp.int32)
    prompt_token_mask = jnp.zeros((row_count, prompt_width), dtype=bool)
    sample_rewards: list[float] = []
    group_ids: list[int] = []
    sample_ids: list[int] = []
    for row_index, (sample, tokens, logprobs, prompt_tokens) in enumerate(rows):
        token_length = len(tokens)
        prompt_length = len(prompt_tokens)
        token_ids = token_ids.at[row_index, :token_length].set(
            jnp.asarray(tokens, dtype=jnp.int32)
        )
        old_logprobs = old_logprobs.at[row_index, :token_length].set(
            jnp.asarray(logprobs, dtype=jnp.float32)
        )
        token_mask = token_mask.at[row_index, :token_length].set(True)
        advantages = advantages.at[row_index, :token_length].set(
            jnp.asarray(sample.advantage, dtype=jnp.float32)
        )
        prompt_token_ids = prompt_token_ids.at[row_index, :prompt_length].set(
            jnp.asarray(prompt_tokens, dtype=jnp.int32)
        )
        prompt_token_mask = prompt_token_mask.at[row_index, :prompt_length].set(True)
        sample_rewards.append(sample.total_reward)
        group_ids.append(group_index_by_id[sample.group_id])
        sample_ids.append(sample.sample_id)
    token_batch = ExternalGrpoTokenBatch(
        token_ids=token_ids,
        old_logprobs=old_logprobs,
        token_mask=token_mask,
        advantages=advantages,
        prompt_token_ids=prompt_token_ids,
        prompt_token_mask=prompt_token_mask,
        sample_rewards=jnp.asarray(sample_rewards, dtype=jnp.float32),
        group_ids=jnp.asarray(group_ids, dtype=jnp.int32),
        sample_ids=jnp.asarray(sample_ids, dtype=jnp.int32),
    )
    token_batch.validate()
    return token_batch


def summarize_external_grpo_batch(batch: ExternalGrpoBatch) -> dict[str, object]:
    """Return compact JSON-safe metrics for logging and dashboards."""
    advantages = [
        sample.advantage for group in batch.groups for sample in group.samples
    ]
    rewards = [
        sample.total_reward for group in batch.groups for sample in group.samples
    ]
    steps = [
        step
        for group in batch.groups
        for sample in group.samples
        for step in sample.replay.steps
    ]
    action_counts = Counter(step.action_label for step in steps)
    action_rewards: dict[str, list[float]] = defaultdict(list)
    for step in steps:
        action_rewards[step.action_label].append(float(step.reward))
    action_reward_mean = {
        action: sum(values) / len(values)
        for action, values in sorted(action_rewards.items())
    }
    step_count = len(steps)
    invalid_count = sum(
        int(bool(step.invalid_format or step.unknown_action or step.masked_action))
        for step in steps
    )
    token_lengths = [
        len(step.token_ids)
        for step in steps
        if step.token_ids is not None
    ]
    prompt_lengths = [
        len(step.prompt_token_ids)
        for step in steps
        if step.prompt_token_ids is not None
    ]
    completion_lengths = [len(step.raw_completion) for step in steps]
    return {
        "schema": batch.schema,
        "group_count": len(batch.groups),
        "sample_count": batch.sample_count,
        "step_count": step_count,
        "mean_reward": batch.mean_reward,
        "min_reward": min(rewards),
        "max_reward": max(rewards),
        "reward_std": _population_std(rewards),
        "advantage_std": _population_std(advantages),
        "mean_abs_advantage": sum(abs(value) for value in advantages) / len(advantages),
        "action_counts": dict(sorted(action_counts.items())),
        "action_distribution": _distribution(action_counts),
        "action_entropy_nats": _entropy_nats(action_counts),
        "unique_action_count": len(action_counts),
        "top_actions": _top_actions(action_counts),
        "action_reward_mean": action_reward_mean,
        "fallback_rate": _rate(sum(int(step.fallback_used) for step in steps), step_count),
        "invalid_action_rate": _rate(invalid_count, step_count),
        "invalid_format_rate": _rate(sum(step.invalid_format for step in steps), step_count),
        "unknown_action_rate": _rate(sum(step.unknown_action for step in steps), step_count),
        "masked_action_rate": _rate(sum(step.masked_action for step in steps), step_count),
        "terminated_rate": _rate(sum(int(step.terminated) for step in steps), step_count),
        "truncated_rate": _rate(sum(int(step.truncated) for step in steps), step_count),
        "mean_episode_length": step_count / batch.sample_count,
        "mean_generated_tokens_per_step": _mean(token_lengths),
        "mean_prompt_tokens_per_step": _mean(prompt_lengths),
        "mean_completion_chars_per_step": _mean(completion_lengths),
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


def _mean(values: Sequence[float | int]) -> float:
    """Return a JSON-safe mean for optional metric collections."""
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def _population_std(values: Sequence[float]) -> float:
    """Return population standard deviation for compact batch diagnostics."""
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def _rate(count: int, total: int) -> float:
    """Return a stable zero-safe rate."""
    return 0.0 if total == 0 else float(count / total)


def _distribution(counts: Counter[str]) -> dict[str, float]:
    """Return action probabilities sorted by label."""
    total = sum(counts.values())
    if total == 0:
        return {}
    return {
        action: float(count / total)
        for action, count in sorted(counts.items())
    }


def _entropy_nats(counts: Counter[str]) -> float:
    """Return categorical entropy over selected action labels."""
    probabilities = _distribution(counts).values()
    return float(-sum(probability * math.log(probability) for probability in probabilities))


def _top_actions(counts: Counter[str], *, limit: int = 5) -> list[dict[str, object]]:
    """Return most frequently selected actions with count and probability."""
    total = sum(counts.values())
    if total == 0:
        return []
    return [
        {
            "action": action,
            "count": count,
            "probability": float(count / total),
        }
        for action, count in counts.most_common(limit)
    ]


jax.tree_util.register_dataclass(
    ExternalGrpoTokenBatch,
    data_fields=[
        "token_ids",
        "old_logprobs",
        "token_mask",
        "advantages",
        "prompt_token_ids",
        "prompt_token_mask",
        "sample_rewards",
        "group_ids",
        "sample_ids",
    ],
    meta_fields=[],
)
