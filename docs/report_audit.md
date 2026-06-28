# Внешний архитектурный аудит: hybrid rollout и PPO-ready контракт

Этот документ переносит присланный внешний аудит в текущую архитектуру проекта. Главный вывод
аудита принят: **реальный LLM-RL rollout нельзя строить вокруг `jax.lax.scan` с динамически
растущей текстовой историей**. `scan` остаётся reference/CPU/JAX-contract слоем, а production
контур должен быть гибридным:

```text
host orchestrator:
  prompt history -> Tunix actor generation -> actor token logprobs -> critic values

accelerator/JAX boundary:
  jax.jit(jax.vmap(CrafTextAdapter.step))(keys, states, action_ids)
```

## Что принято из аудита

1. `tunix_craftext.rollouts.reference.collect_rollout_scan*` не считается production LLM-RL collector.
   Он остаётся reference-контрактом для fixed-shape tests и CPU/JAX parity.
2. Добавлен отдельный слой `tunix_craftext.rollouts.hybrid`, где PPO-ready step явно содержит:
   `prompt_tokens`, `generation_tokens`, `actor_log_probs`, `values`,
   `generation_token_mask`, `step_mask` и optional `action_mask`.
3. PPO actor loss должен маскироваться по двум осям:
   generated-token mask исключает padding ответа модели, а step mask исключает post-terminal
   строки батча.
4. Post-hoc fallback в `batched_rollout` остаётся safety/evidence механизмом, но не должен быть
   финальным production способом action masking. Целевой путь — action-mask/logits-processor
   на стороне Tunix sampling, чтобы модель не могла сэмплировать запрещённый action token.
5. `batched_rollout.py` остаётся полезным synchronous precursor: host prompt/render/LLM/decode
   плюс `jax.vmap(adapter.step)`. Но для PPO training он должен конвертироваться в hybrid
   trajectory или Tunix agentic trajectory с token logprobs/values/masks.

## Что скорректировано относительно аудита

Внешний аудит формулирует PPO как главный целевой стек. В текущей roadmap проекта production
порядок остаётся:

1. **Tunix Agentic GRPO** как первый golden path, потому что он не требует trainable critic и
   быстрее проверяет весь CrafText/ToolAgent/RLCluster transport.
2. **Agentic PPO** как extension lane поверх того же transport, где уже обязателен critic,
   values, returns, old logprobs, policy version и checkpoint actor/critic/reference.

То есть мы не возвращаем старый локальный PPO stack в production. Hybrid rollout contract
добавлен как общий data boundary для PPO, PPO-Lag, CPO и связанных notebooks/tests.

## Текущий кодовой результат

- `tunix_craftext.rollouts.hybrid.HybridPpoStep` — один PPO-ready батчевый шаг.
- `tunix_craftext.rollouts.hybrid.HybridPpoTrajectory` — tuple шагов + stacked `step_masks`
  формы `[T, B]`.
- `hybrid_step_from_text_trajectory()` — adapter из replay-derived `TextTrajectoryBatch` в
  `HybridPpoStep`; он сохраняет generated-token evidence, исключает fallback rows из
  `policy_token_mask` и вычисляет alive-before-step mask.
- `last_valid_token_values()` — bridge из token critic values `[B, L]` в step values `[B]`.
- `compute_masked_step_token_ppo_loss()` — TDD primitive, который доказывает, что PPO actor loss
  игнорирует generated-token padding и post-terminal строки батча.
- `tests/unit/test_hybrid_rollout.py` покрывает shape contract, stacked masks и masked PPO loss.
- Notebooks 10/11/12 переписаны под этот setup: replay, batched Qwen rollout и real Gemma
  actor+critic scoring теперь проходят через `HybridPpoStep` без deferred TODO-блоков.

## Открытые задачи

- Подключить hybrid trajectory adapter к реальному Tunix `TrajectoryCollectEngine` /
  `AgenticPPOLearner` evidence path.
- Добавить model-side Tunix logits processor/action masker вместо post-hoc fallback.
- Добавить hardware-gated one-update PPO test: actor и critic weights меняются, checkpoints
  actor/critic/reference сохраняются, resume parity проходит на fixed fixture.
- Расширить `AgenticPPOTrainExample` до cost critic для PPO-Lag/CPO после прохождения базового
  PPO gate.
