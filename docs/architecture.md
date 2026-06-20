# Архитектура

```mermaid
flowchart LR
  E[CrafText / CagedCrafText\nvendored baseline] --> A[Environment adapter]
  P[MegaPrompts\nvendored assets] --> A
  A --> R[Rollout collector\nJAX scan]
  R --> C[Trajectory contract\nT × B]
  C --> G[GAE / returns / masks]
  G --> L[Algorithm: PPO → GRPO\nloss functions]
  L --> O[Flax TrainState + Optax]
  O --> T[Tunix bridge\nmodel / sampling APIs]
  O --> K[Orbax checkpoints]
  C --> M[Metrics + trajectory artifacts]
  K --> D[Docs site + Git provenance]
  M --> D
```

## Слои и запреты

| Слой | Ответственность | Не имеет права знать |
| --- | --- | --- |
| `vendor/` | снимок CrafText, CagedCrafText, MegaPrompts | алгоритм обучения |
| `adapters/` | нормализовать reset/step, action mask, text/token encoding | PPO/GRPO loss |
| `rollout/` | batched scan, RNG partitioning, `Transition` | filesystem, логгер |
| `algorithms/` | GAE, clipped loss, KL/reward shaping | конкретная среда |
| `training/` | Flax state, Optax update, Tunix bridge | формат prompt YAML |
| `checkpointing/` | Orbax restore/save и schema version | env step loop |
| `reporting/` | metrics, benchmark JSON, docs provenance | мутабельный trainer state |

## Контракт траектории

`Transition` и `RolloutBatch` — главная точка совместимости. Все обязательные поля
батчированы, rollout time-major: observation/action/reward/terminated/truncated/
log_prob/value имеют первую пару осей `[T, B]`, а `bootstrap_value` — `[B]`.

Terminal state отделён от truncation: GAE маскирует настоящий terminal, а timeout
может bootstrap-иться по выбранной политике. Это решение тестируется отдельными
табличными примерами до первой оптимизации.

## CrafText adapter boundary

`CrafTextAdapter` и `CagedCrafTextAdapter` принимают vendor `reset(key, params)` и
`step(key, state, action, params)`, но отдают training-safe `EnvironmentReset` и
`EnvironmentStep`. Они фиксируют action mask как `[A]`, переводят vendor `done` в
`terminated` и создают явный all-false `truncated`: текущий vendor API не различает
timeout. Vendor `info` не протекает дальше boundary.

Contract golden fixture запускает два детерминированных CrafText-shaped env на 8-step
траекториях; real CrafText/Caged smoke test запускается только с extra `envs`. Следующий
шаг — заменить fixture на tiny world preset parity при установленном Craftax.

## Compiled rollout collection

`collect_rollout_scan` использует `jax.lax.scan` для фиксированного horizon и возвращает тот же
`RolloutBatch` `[T, B, ...]`, что и читаемый reference collector. Контракты зарегистрированы как
JAX PyTrees, поэтому collector можно оборачивать в `jax.jit`. Любой новый policy/env сначала
сверяется leaf-by-leaf с `collect_rollout`; compilation/warmup измеряется отдельно от steady state.

`Transition.done` использует `jax.numpy`, а `RolloutBatch.validate()` намеренно остаётся
host-side boundary-проверкой: она читает только static `shape` metadata и валидирует ведущие
оси каждой leaf во вложенных observation/action PyTrees. Регистрация PyTree находится в
`contracts.py`, а не у конкретного collector-а, поэтому контракт одинаково безопасен для
reference, scan и будущих learner-модулей.

Численные поля публичных `Transition` и `RolloutBatch` аннотированы как `jax.Array`:
после boundary они уже нормализованы и пригодны для JIT. Reference policy/step могут вернуть
`jax.typing.ArrayLike`, но collector сразу приводит такие значения в `jax.Array`. Небыстрый,
не-JIT reference collector намеренно stack-ает на host и нормализует один раз; compiled path
остаётся целиком JAX. Это допускает NumPy fixture в тесте, не размывая runtime-контракт trainer-а.

## Tunix как расширяемая граница

Tunix не должен становиться скрытой внутренней зависимостью. `TunixPolicyAdapter`
получит три небольших операции: `sample`, `log_prob_and_value`, `apply_gradients`.
Версия Tunix, сигнатуры и функциональный parity-smoke тест фиксируются в
`compatibility/tunix.yaml`. Изменение API — отдельный ADR и compatibility PR.
