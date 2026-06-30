# ADR 0006: явная граница rollout generation для Tunix Qwen

## Статус

Принято.

## Контекст

Локальный Agentic GRPO профиль использует Qwen 2.5 0.5B и Tunix `RLCluster` с ролями
`actor`, `rollout`, `reference`. Текущая топология `qwen_agentic_grpo_local.yaml`
создаёт mesh с осями `fsdp,tp` даже на одном устройстве:

```yaml
axis_name: fsdp,tp
roles:
  actor: [0]
  rollout: [0]
  reference: [0]
```

Static divisibility checks проходят: `num_heads`, `embed_dim`, `vocab_size` делятся на
degree role mesh. Но реальный `GRPOLearner.train()` падает внутри Tunix Qwen sampler на
embedding gather:

```text
Use .at[...].get(out_sharding=...) or explicitly pass sharding to jaxpr.
```

Это означает, что проблема не в `asyncio`, HF snapshot или `skip_jit`. Проблема в том, что
generation path получает Qwen параметры с sharding-annotation `fsdp,tp`, а embedding lookup
в текущем upstream Tunix Qwen path не имеет совместимой явной gather/sharding boundary.

Мы сверили это с двумя внешними практиками:

- `chuyishang/jax-lm` валидирует semantic sharding mode отдельно от divisibility:
  `tp` и `fsdp_tp` имеют реальные требования к mesh, а unsharded/single-device inference
  не маскируется под symbolic tensor parallel.
- NVIDIA JAX-Toolbox/JAX inference offloading разделяет trainer и rollout engine:
  generation обслуживается inference backend-ом с явным mapping/resharding boundary,
  а не случайным переиспользованием trainer-sharded model object.

## Решение

1. `validate_agentic_grpo_preflight()` становится backend-aware. Он больше не проверяет
   только размеры тензоров и batch divisibility; он также отклоняет известную битую связку:
   Qwen + `vanilla-jax-sharded` rollout + `fsdp,tp` mesh.
2. Evidence/scripted checks остаются разрешены. Они нужны для проверки CrafText tasks,
   MegaPrompts, ToolAgent, logging, checkpoint/evidence directories и validation artifacts
   без heavy model allocation.
3. Реальный Qwen Agentic GRPO через Tunix vanilla rollout считается заблокированным до
   одного из следующих implementation lanes:
   - `single-device-jax`: отдельный unsharded generation sampler/adapter для локального
     one-GPU bring-up;
   - `vllm-offload`: production-style rollout engine по паттерну NVIDIA JAX-Toolbox;
   - upstream Tunix fix: Qwen embedding gather получает корректный `out_sharding`/layout
     contract для sharded generation.
4. Runner и notebooks не должны silently запускать broken path. Они должны показывать
   понятный preflight blocker до загрузки весов.

## Последствия

- Мы перестаём путать “профиль делится по размерам” с “generation backend реально работает”.
- Локальные notebooks остаются полезны для ручного поднятия среды, scripted GRPO smoke,
  task sampling, evidence и object construction, но heavy `learner.train()` должен быть
  gated до готового rollout backend-а.
- Следующий production slice — не ещё один костыль вокруг `skip_jit`, а реализация явного
  rollout backend boundary и hardware-gated integration test.
