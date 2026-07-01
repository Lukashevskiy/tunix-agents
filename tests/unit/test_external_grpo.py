"""External vLLM rollout to GRPO evidence tests."""

from __future__ import annotations

import json

import jax.numpy as jnp
import pytest

from tunix_craftext.adapters import CrafTextAdapter
from tunix_craftext.artifacts.replay import ReplayArtifact, ReplayStep
from tunix_craftext.env.prompts import ActionCatalog, PromptContext, RenderedPrompt
from tunix_craftext.models.llm import LlmResponse
from tunix_craftext.rollouts.batched import collect_batched_text_rollout
from tunix_craftext.training.external_grpo import (
    ExternalGrpoError,
    external_grpo_batch_from_batched_rollout,
    external_grpo_batch_from_replays,
    external_grpo_group_from_replays,
    group_normalized_advantages,
    save_external_grpo_batch,
    summarize_external_grpo_batch,
    token_batch_from_external_grpo,
)


class _Environment:
    def reset(self, key, params):
        del key, params
        return jnp.asarray(0), jnp.asarray(0)

    def step(self, key, state, action, params):
        del key, params
        next_state = state + action
        return next_state, next_state, jnp.asarray(action, dtype=jnp.float32), False, {}


class _Renderer:
    def render(self, context: PromptContext[object]) -> RenderedPrompt:
        return RenderedPrompt(f"state={context.observation}", context.actions, "test")


class _TokenBackend:
    def complete(self, request):
        return self.complete_batch((request,))[0]

    def complete_batch(self, requests):
        responses = []
        for index, request in enumerate(requests):
            action = "DO" if index % 2 else "NOOP"
            action_id = 2 + index
            responses.append(
                LlmResponse(
                    f"<action>{action}</action>",
                    "fake-vllm",
                    "fake-qwen",
                    token_ids=(action_id,),
                    token_logprobs=(-0.1 * (index + 1),),
                    prompt_token_ids=tuple(range(1, len(request.prompt.text.split()) + 2)),
                )
            )
        return tuple(responses)


def _replay(total_reward: float, *, token_evidence: bool = True) -> ReplayArtifact:
    token_ids = (11, 12) if token_evidence else None
    token_logprobs = (-0.1, -0.2) if token_evidence else None
    prompt_token_ids = (1, 2, 3) if token_evidence else None
    return ReplayArtifact(
        config_path="configs/env/text/qwen_craftext.yaml",
        commit="abc123",
        backend="vllm-offload",
        steps=(
            ReplayStep(
                index=0,
                prompt="goal",
                raw_completion="<action>NOOP</action>",
                action_id=0,
                action_label="NOOP",
                reward=total_reward,
                terminated=False,
                token_ids=token_ids,
                token_logprobs=token_logprobs,
                prompt_token_ids=prompt_token_ids,
            ),
        ),
    )


def test_group_normalized_advantages_are_zero_mean_and_stable_for_ties() -> None:
    """External GRPO uses per-task reward normalization and handles no-signal groups."""
    advantages = group_normalized_advantages((1.0, 2.0, 3.0))

    assert sum(advantages) == pytest.approx(0.0, abs=1e-6)
    assert advantages[0] < 0.0 < advantages[-1]
    assert group_normalized_advantages((2.0, 2.0)) == (0.0, 0.0)


def test_external_grpo_group_requires_multiple_replays_and_token_provenance() -> None:
    """Trainer-facing evidence must not silently drop token/logprob provenance."""
    with pytest.raises(ExternalGrpoError, match="at least two"):
        external_grpo_group_from_replays(
            goal="collect wood",
            group_id="wood-0",
            replays=(_replay(1.0),),
        )

    with pytest.raises(ExternalGrpoError, match="lacks generated token ids"):
        external_grpo_group_from_replays(
            goal="collect wood",
            group_id="wood-0",
            replays=(_replay(1.0, token_evidence=False), _replay(2.0)),
        )


def test_external_grpo_batch_chunks_replays_and_summarizes_metrics() -> None:
    """Ordered external vLLM replays become stable GRPO groups and summary metrics."""
    batch = external_grpo_batch_from_replays(
        goal="collect wood",
        group_prefix="wood",
        group_size=2,
        replays=(_replay(1.0), _replay(3.0), _replay(2.0), _replay(2.0)),
    )

    assert batch.sample_count == 4
    assert [group.group_id for group in batch.groups] == ["wood-0", "wood-1"]
    assert [sample.advantage for sample in batch.groups[0].samples] == pytest.approx(
        [-0.999999, 0.999999],
        rel=1e-5,
    )
    assert [sample.advantage for sample in batch.groups[1].samples] == [0.0, 0.0]
    assert summarize_external_grpo_batch(batch)["group_count"] == 2


def test_external_grpo_batch_from_batched_rollout_preserves_replay_evidence() -> None:
    """The direct vLLM rollout path can feed GRPO grouping without Tunix RLCluster."""
    adapter = CrafTextAdapter(_Environment(), None, action_count=2)
    rollout = collect_batched_text_rollout(
        adapter,
        _Renderer(),
        _TokenBackend(),
        actions=ActionCatalog(("NOOP", "DO")),
        batch_size=2,
        horizon=1,
        seed=0,
        goal="collect wood",
        max_new_tokens=4,
        invalid_action="fallback",
        fallback_action_id=0,
    )

    batch = external_grpo_batch_from_batched_rollout(
        rollout,
        goal="collect wood",
        group_prefix="wood",
        group_size=2,
        config_path="configs/env/text/qwen_craftext.yaml",
        commit="abc123",
        backend="fake-vllm",
    )

    assert batch.sample_count == 2
    assert [sample.total_reward for sample in batch.groups[0].samples] == [0.0, 1.0]
    assert batch.groups[0].samples[0].replay.steps[0].token_ids == (2,)
    assert batch.groups[0].samples[1].replay.steps[0].prompt_token_ids is not None


def test_save_external_grpo_batch_writes_stable_json(tmp_path) -> None:
    """GRPO evidence can be handed to local/Comet artifact loggers as one file."""
    batch = external_grpo_batch_from_replays(
        goal="collect wood",
        group_prefix="wood",
        group_size=2,
        replays=(_replay(1.0), _replay(3.0)),
    )
    path = tmp_path / "external-grpo.json"

    save_external_grpo_batch(path, batch)

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "tunix-craftext.external-grpo/v1"
    assert payload["groups"][0]["samples"][0]["replay"]["schema"] == "tunix-craftext.replay/v3"


def test_token_batch_from_external_grpo_broadcasts_group_advantages_to_tokens() -> None:
    """External GRPO evidence becomes token rows consumable by GRPO loss."""
    batch = external_grpo_batch_from_replays(
        goal="collect wood",
        group_prefix="wood",
        group_size=2,
        replays=(_replay(1.0), _replay(3.0)),
    )

    token_batch = token_batch_from_external_grpo(batch)

    assert token_batch.token_ids.shape == (2, 2)
    assert token_batch.prompt_token_ids.shape == (2, 3)
    assert token_batch.token_mask.tolist() == [[True, True], [True, True]]
    assert token_batch.sample_rewards.tolist() == [1.0, 3.0]
    assert token_batch.group_ids.tolist() == [0, 0]
    assert token_batch.sample_ids.tolist() == [0, 1]
    assert bool(
        jnp.allclose(
            token_batch.advantages,
            jnp.asarray([[-0.999999, -0.999999], [0.999999, 0.999999]]),
            rtol=1e-5,
        )
    )
