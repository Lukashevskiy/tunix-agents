# Подробный план реализации

## 0. Базовая линия — сделано в этом каркасе

- [x] Отдельный git-репозиторий и vendor snapshot CrafText, CagedCrafText, MegaPrompts.
- [x] Лицензии исходных пакетов сохранены внутри vendor.
- [x] Нейтральный контракт trajectory и тестируемый reference collector.
- [x] Unit, integration и performance test lanes; MkDocs site и ADR.

## 1. Совместимость и измеримая baseline

1. Зафиксировать Python/JAX/JAXLIB/Tunix/Flax/Optax/Orbax в lockfile для CPU и целевого
   accelerator, с отдельной таблицей compatibility.
2. Вынести exact source revision и SHA256 vendor-снимков в `vendor/manifest.json`.
3. Написать `CrafTextAdapter.reset/step`, который возвращает статические pytree и
   action-mask; сделать Caged-вариант тем же протоколом.
4. Создать deterministic tiny-world fixture: seed, 2 env, 8 steps, золотая
   последовательность reward/done/allowed actions.
5. Снять baseline: reset/step throughput, host→device transfer, compile time, peak HBM,
   tokens/s и env-steps/s. Данные — JSON в `artifacts/benchmarks/`, не только console log.

**Gate:** две среды проходят parity fixture; baseline имеет hardware, commit, config и seed.

## 2. Pure JAX collection

1. Реализовать `lax.scan` collector с split PRNG per update/env/action.
2. Добавить done-reset semantics без Python ветвления; отдельно проверить terminated vs
   truncated и action mask.
3. Сравнить reference collector с JIT collector на fixed fixture leaf-by-leaf.
4. Добавить sharding API заранее (`Mesh`/named axes), но начать с одного device.
5. Профилировать compilation отдельно от steady-state и документировать warmup.

**Gate:** exact parity дискретных полей, численная tolerance для float, не хуже baseline
по env-step/s после warmup.

## 3. Алгоритмический минимум: PPO

1. TDD для discounted return, GAE, advantage normalization, masks и value bootstrap.
2. Чистые функции loss: policy clip, value clip, entropy, KL; каждая имеет hand-computed
   mini-batch test.
3. Flax actor-critic и `TrainState` с Optax schedule/gradient clipping.
4. Один update на synthetic trajectory → loss finite, params change, checkpoint round-trip.
5. Запустить tiny CrafText end-to-end и сохранить trajectory/rendered prompt/metrics.

**Gate:** loss tests, deterministic smoke learning, Orbax resume даёт идентичный следующий update.

## 4. Tunix bridge и LLM policy

1. Зафиксировать проверенный Tunix release и написать adapter только по его публичному API.
2. Унифицировать tokenizer/action decoder: invalid action → observable, metric и controlled
   fallback, никогда не silent coercion.
3. Реализовать sampling/logprob/value bridge; проверить parity против прямого вызова Tunix.
4. Сделать SFT warm-start и PPO на коротких fixed prompt rollouts.
5. Лишь после этого добавить GRPO как самостоятельный algorithm module, не как fork PPO.

**Gate:** prompt→sample→environment→loss интеграционный тест, стабильно записываемый replay.

## 5. Масштабирование, отчётность и release

1. Добавить multi-device tests с порогами масштабирования и явной degradation report.
2. Добавить preemption/resume test и schema migration checkpoint.
3. CI: lint + unit на CPU; integration env lane; nightly perf with comparison against baseline.
4. Генерировать docs site из config schema, benchmark JSON, git revision и Mermaid/ADR.
5. Release checklist: reproducibility card, known limitations, performance table, migration guide
   для конфигов VERL.

## Текущая ветка реализации

`foundation/contracts-and-docs`: vendor snapshot, contracts, reference rollout, test lanes,
documentation site. Следующий маленький PR: **CrafTextAdapter + golden tiny-world fixture**.
