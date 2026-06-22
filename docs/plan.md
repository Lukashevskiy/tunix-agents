# Подробный план реализации

## 0. Базовая линия — сделано в этом каркасе

- [x] Отдельный git-репозиторий и vendor snapshot CrafText, CagedCrafText, MegaPrompts.
- [x] Лицензии исходных пакетов сохранены внутри vendor.
- [x] Нейтральный контракт trajectory и тестируемый reference collector.
- [x] JAX-native trajectory contracts: PyTree registration рядом с типами, JIT-safe terminal mask
  и полная shape-проверка вложенных observation/action leaves на host boundary.
- [x] Unit, integration и performance test lanes; MkDocs site и ADR.

## 1. Совместимость и измеримая baseline

- [~] Зафиксировать Python/JAX/JAXLIB/Tunix/Flax/Optax/Orbax в lockfile для CPU и целевого
   accelerator, с отдельной таблицей compatibility.
- [ ] Вынести exact source revision и SHA256 vendor-снимков в `vendor/manifest.json`.
- [x] Написать `CrafTextAdapter.reset/step`, который возвращает статические pytree и
   action-mask; сделать Caged-вариант тем же протоколом.
- [ ] Создать deterministic tiny-world fixture: seed, 2 env, 8 steps, золотая
   последовательность reward/done/allowed actions.
- [~] Снять baseline: reset/step throughput, host→device transfer, compile time, peak HBM,
   tokens/s и env-steps/s. Данные — JSON в `artifacts/benchmarks/`, не только console log.

**Gate:** две среды проходят parity fixture; baseline имеет hardware, commit, config и seed.

## 2. Pure JAX collection

- [~] Реализовать `lax.scan` collector с split PRNG per update/env/action.
- [ ] Добавить done-reset semantics без Python ветвления; отдельно проверить terminated vs
   truncated и action mask.
- [ ] Сравнить reference collector с JIT collector на fixed fixture leaf-by-leaf.
- [ ] Добавить sharding API заранее (`Mesh`/named axes), но начать с одного device.
- [~] Профилировать compilation отдельно от steady-state и документировать warmup.

**Gate:** exact parity дискретных полей, численная tolerance для float, не хуже baseline
по env-step/s после warmup.

## 3. Алгоритмический минимум: PPO

- [~] TDD для discounted return, GAE, advantage normalization, masks и value bootstrap.
- [~] Чистые функции loss: policy clip, value clip, entropy, KL; каждая имеет hand-computed
   mini-batch test.
- [~] Flax actor-critic и `TrainState` с Optax schedule/gradient clipping.
- [x] Один update на synthetic trajectory → loss finite, params change, checkpoint round-trip.
- [ ] Запустить tiny CrafText end-to-end и сохранить trajectory/rendered prompt/metrics.

**Gate:** loss tests, deterministic smoke learning, Orbax resume даёт идентичный следующий update.

## 4. Tunix bridge и LLM policy

- [~] Зафиксировать проверенный Tunix release и написать adapter только по его публичному API.
- [~] Унифицировать tokenizer/action decoder: invalid action → observable, metric и controlled
  fallback, никогда не silent coercion.
- [~] Добавить Qwen local smoke через публичный Tunix sampler; single-device backend не является
  production multi-device path.
- [~] Реализовать Qwen chat-template и sampling/logprob/value bridge: chat-template, sampler
  cache sizing, raw token logprobs и invalid-action/fallback replay уже проверены; остаются
  value bridge и output-parity fixture против прямого Tunix call.
- [ ] Построить workload path через Tunix `RLCluster` и versioned `role_to_mesh` для
  actor/rollout/critic/reference; не реализовывать собственный GPU scheduler.
- [ ] Сделать SFT warm-start и PPO на коротких fixed prompt rollouts.
- [ ] Лишь после этого добавить GRPO как самостоятельный algorithm module, не как fork PPO.

**Gate:** prompt→sample→environment→loss интеграционный тест, стабильно записываемый replay.

## 4a. Model interoperability

- [x] Базовый versionable template для state_dict → JAX/Flax PyTree и безопасный LoRA merge.
- [ ] Добавить архитектурные templates для выбранных Tunix-compatible моделей с output parity fixtures.
- [ ] Добавить QLoRA/dequantization adapter с numerical tolerance tests.
- [~] Добавить Orbax import/export и round-trip test model + optimizer + adapter metadata.

## 5. Масштабирование, отчётность и release

- [ ] Добавить multi-device tests с порогами масштабирования и явной degradation report.
- [ ] Добавить preemption/resume test и schema migration checkpoint.
- [ ] CI: lint + unit на CPU; integration env lane; nightly perf with comparison against baseline.
  MVP Python 3.11–3.13 matrix запускается отдельно на release tag `mvp-v*`.
- [x] Генерировать docs site из config schema, benchmark JSON, git revision и Mermaid/ADR.
- [ ] Release checklist: reproducibility card, known limitations, performance table, migration guide
   для конфигов VERL.

## 5a. Расширяемость и исследовательская наблюдаемость

- [~] Ввести typed `AlgorithmSpec`/registry: advantage transform, pure loss и immutable metrics;
  PPO становится первой реализацией без изменения orchestration transport contracts.
- [ ] Добавить будущий GRPO/GSPO только по первичному источнику с отдельным group-shape contract
  `[B, G, T]`, hand-computed fixtures и explicit reference log-probs.
- [~] Ввести `ModelAdapter`/profile boundary: Qwen local loader и tokenizer уже отделены от core;
  остаются chat-template, `RLCluster` profile и output parity fixture.
- [ ] Экспортировать structured JAX metrics на epoch boundary в JSONL и optional TensorBoard.
- [ ] Добавить явный profiler command: warmup, ограниченный trace, Perfetto artifact, compile
  ledger с shape/dtype/mesh fingerprint и recompilation alert.
- [ ] Добавить Chex assertions только в host boundary/unit lanes, без blocking внутри JIT loops.

**Gate:** новый algorithm/model подключается registry entry и fixtures; JIT transport loop не
меняется, а trace и metrics связываются с config/commit/mesh provenance.

## Текущая ветка реализации

`main`: CrafText/Caged adapter boundary и compiled `lax.scan` collector готовы; scan
parity/steady-state benchmark проходят. Qwen 2.5 0.5B локально загружается через публичный
Tunix API в single-device smoke-профиле. Следующий implementation slice: **Qwen chat-template
и token/logprob bridge**, затем hardware-gated `RLCluster` profile с явным `role_to_mesh`.
