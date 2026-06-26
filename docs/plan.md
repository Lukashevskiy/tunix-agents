# Golden Pipeline: Tunix Agentic GRPO + CrafText

## Решение

Проект прекращает развивать несколько равноправных training paths. Единственный
product-critical путь использует публичный Tunix 0.1.7 Agentic RL API:

```text
task/seed -> CrafText Agentic environment -> ToolAgent -> RLCluster rollout
-> multi-turn trajectories -> Agentic GRPO -> checkpoints/metrics -> evaluation
```

Алгоритм первого полного pipeline - **Agentic GRPO**. В закреплённом Tunix это
готовый agentic learner с `TrajectoryCollectEngine` и `RolloutOrchestrator`.
Он не требует отдельного trainable critic, поэтому не следует сначала строить
локальный PPO/value-head путь, который не связан с model policy.

Локальные `algorithms.py`, `learner.py`, Flashbax и token-PPO остаются
исследовательскими компонентами. Они не являются production training loop и не
должны получать новые feature-изменения, пока golden pipeline не завершён.

## Definition Of Done

Один CLI command, получивший versioned YAML и local/remote model profile,
должен:

1. создать trainable actor, rollout actor и frozen reference через `RLCluster`;
2. собрать минимум две multi-turn CrafText траектории на один task через
   `ToolAgent` и `craftext_step(action=...)`;
3. выполнить хотя бы один Agentic GRPO update и доказать изменение actor weights;
4. записать checkpoint, trajectory JSONL, metrics JSONL и evaluation artifact;
5. восстановить checkpoint и воспроизвести следующий update на fixed fixture;
6. пройти CPU/fake-model contract tests и hardware-gated real-model integration test.

Нельзя считать pipeline готовым по notebook, standalone sampler или по одному
`ClusterConfig`: DoD требует фактический `RLCluster`, learner и update.

## Порядок исполнения после архитектурного аудита

Следующие этапы исполняются строго по порядку. Они не создают второй PPO-first
roadmap: первый production path — Agentic GRPO с ролями `actor`, `rollout` и
`reference`; trainable critic теперь оформлен отдельным Agentic PPO extension
lane поверх upstream `AgenticRLLearner`, но его hardware one-update gate не
смешивается с GRPO acceptance gates.

| Очередь | Вертикальный срез | Acceptance gate | Не делать до gate |
| --- | --- | --- | --- |
| 0 | Hygiene и task semantics | `ruff`, `mypy`, unit suite; real CrafText test доказывает, что prompt содержит и user goal, и scenario instruction, world preset и Caged constraint | Не менять learner/mesh ради обхода неясного task contract |
| 1 | Deterministic golden fixture | 2 task groups × 2 generations × 8 turns, fixed seeds; expected tool calls/rewards/done/action masks экспортированы как versioned fixture | Не заявлять multi-turn environment production-ready |
| 2 | Reproducible GRPO profile | `configs/grpo/qwen_agentic_local.yaml` содержит run/model/topology/workload/evidence paths; runner пишет profile/vendor SHA256, model revision/licence, config hash и package versions рядом с run | Не загружать weights по незафиксированному profile |
| 2.5 | Full CLI orchestration layer | `tunix-craftext profile/verify/train/...` спроектирован как thin orchestration layer; первый implementation slice ограничен profile + verify без heavy imports | Не переносить business logic в CLI и не ломать scripts до wrapper migration |
| 2.7 | Hybrid PPO rollout contract | `hybrid_rollout.py` хранит actor token logprobs, critic values, generation-token masks и step masks; `rollout.py` явно остаётся reference/fixed-shape слоем | Не использовать `lax.scan` collectors как production LLM-RL rollout |
| 3 | Real RLCluster rollout | Accelerator-gated fixture создаёт actor/rollout/reference, проверяет role mesh и actor–rollout token/logprob parity | Не мерить distributed throughput и не строить custom scheduler |
| 4 | One Agentic GRPO update | Один `GRPOLearner` update меняет actor weights, loss конечен, generation groups валидны, metrics включают return/success/invalid-action/KL | Не помечать model factory или runner готовыми |
| 5 | Evidence, resume, evaluation | Checkpoint включает learner/cluster policy version; resumed next update совпадает с continuous; fixed evaluation сравнивает actor/reference | Не выпускать “trained checkpoint” или notebook как substitute |
| 6 | CI и performance | PR CPU gate: Ruff + mypy + unit/fake agentic; accelerator smoke и nightly benchmark имеют отдельные runners и threshold evidence | Не выводить scale-up из macOS/CPU результатов |

### Первый implementation slice: task semantics и hygiene

1. Исправить composition prompt: `task.goal` не должен теряться при наличии
   `CrafTextEpisodeContext`; scenario instruction, world preset и text constraint
   остаются явными полями prompt context.
2. Добавить real-runtime integration test на этот prompt и negative test на
   отсутствие/несогласованность scenario context.
3. Устранить текущие Ruff findings и добавить обязательный PR workflow для
   Ruff, mypy и unit/fake-agentic tests.
4. Зафиксировать exact fixture schema и vendor/model provenance contract до
   первого accelerator allocation.

**Первый test-first критерий:** два разных `task.goal` на одном фиксированном
scenario должны давать разные rendered prompts, но одинаковые legal action
catalogue и deterministic initial environment state.

## 0. Audit And Reproducibility Gate

- [~] Добавить SHA256 и exact revisions каждого vendor snapshot в `vendor/manifest.json`:
  manifest exact revision есть, SHA256 фиксируется в per-run evidence; per-component
  snapshot hashes остаются отдельным hardening task.
- [x] Зафиксировать один train profile: Qwen 2.5 0.5B, tokenizer, model revision,
  licence acknowledgement, target accelerator, mesh и memory budget.
- [x] Добавить Qwen mesh preflight: до загрузки весов проверяются head/embed/vocab
  divisibility по role mesh, train/rollout micro-batch и prompt+generation KV cache budget.
- [~] Добавить exact deterministic fixture: fake-agentic fixture покрывает 2 task groups ×
  2 generations × 8 turns, fixed seeds, expected tool calls/rewards/done/action masks;
  real CrafText/Qwen parity остаётся отдельным integration gate.
- [~] Разделить lockfile evidence: Agentic GRPO profile пишет package versions в run
  manifest; отдельный accelerator lockfile lane ещё не подключён.
- [x] Сделать `make verify-golden` обязательным local entrypoint без implicit downloads:
  Ruff, mypy, fake-agentic/profile/RLCluster/preflight unit contracts, task sync,
  docs build и repository audit.

**Gate:** fixture, model profile и dependency provenance проверяются до загрузки весов.

## 0.5. Full CLI Orchestration Layer

- [x] Спроектировать `tunix-craftext` как полновесный, но thin CLI: команды `profile`,
  `env`, `prompt`, `rollout`, `train`, `eval`, `benchmark`, `docs`, `verify`, `audit`;
  доменная логика живёт в use-case modules, не в CLI handlers.
- [ ] Добавить scaffold `tunix_craftext.cli` и console scripts `tunix-craftext`/`tcx`.
- [ ] Реализовать первый TDD slice: `profile validate`, `profile evidence`, `verify golden`;
  `--help` и profile commands не импортируют heavy Qwen/Tunix modules.
- [ ] Перенести `run_agentic_grpo.py` в use-case слой и оставить script wrapper.
- [ ] Затем мигрировать env/prompt/rollout/benchmark/docs/audit команды.

**Gate:** `tunix-craftext profile validate configs/grpo/qwen_agentic_local.yaml` и
`tunix-craftext verify golden` проходят на CPU без downloads и accelerator allocation.

## 0.7. External PPO Audit And Hybrid Rollout Boundary

- [x] Перенести внешний PPO/hybrid rollout аудит в `docs/report_audit.md` и root
  `report_audit.md` pointer; явно отметить, какие рекомендации приняты, а какие
  скорректированы под GRPO-first roadmap.
- [x] Зафиксировать, что `collect_rollout_scan*` — reference/fixed-shape collector для
  CPU/JAX parity, а не production LLM-RL collector с динамической текстовой историей.
- [x] Добавить `tunix_craftext.hybrid_rollout`: `HybridPpoStep`,
  `HybridPpoTrajectory`, stacked `step_masks` и masked token-step PPO loss primitive.
- [x] Покрыть hybrid contract unit tests: shape validation, mismatched token logprobs,
  time-major step masks, generated-token padding и post-terminal rows.
- [ ] Подключить `HybridPpoStep`/trajectory adapter к реальному Tunix Agentic PPO evidence
  path: `TrajectoryCollectEngine`/`AgenticPPOLearner` должны получать old logprobs,
  values, masks и policy version без неявного replay guessing.
- [ ] Реализовать model-side Tunix action masking/logits processor; текущий fallback в
  `batched_rollout.py` остаётся safety/evidence mechanism, а не финальный sampling policy.

**Gate:** один PPO-ready hybrid trajectory из CrafText/Tunix rollout содержит валидные
`actor_log_probs`, `values`, `generation_token_mask`, `step_mask` и проходит masked loss smoke
без вклада post-terminal padding.

## 1. Production Multi-turn Environment

- [x] Заменить nested `build_craftext_agentic_environment()` на module-level,
  serializable `BaseTaskEnv`, который Tunix может создавать из task/config в workers.
- [x] Реализовать task factory: scenario, seed, goal, horizon и group id -> isolated
  CrafText environment; исключить sharing JAX state между параллельными episodes.
- [x] Сделать `craftext_step` единственным ToolAgent tool: strict action schema,
  legal-action validation, observable invalid action, next prompt/tool output.
- [~] Сохранить per-turn CrafText state/action/reward/done и token provenance в Tunix trajectory.
- [~] Подключить реальный Qwen ToolAgent fixture, не только fake `model_call`.

**Gate:** `TrajectoryCollectEngine` собирает fixed 2 x 8 multi-turn fixture и
реальный Qwen smoke без fallback-only trajectory.

## 2. Tunix RLCluster Profile

- [ ] Перевести topology contract с обязательных actor/rollout/critic/reference
  на Agentic GRPO roles: actor, rollout, reference; critic допустим только для
  отдельного Agentic PPO/PPO-Lag/CPO profile.
- [x] Создать `AgenticGrpoWorkloadSpec` и YAML profile с batch sizes, generation
  count, sequence limits, optimizer, checkpoint root, metrics directory и
  pre-allocation evidence manifest.
- [~] Реализовать model factory, которая создаёт actor/reference совместимых
  Qwen model objects и tokenizer для `RLCluster`; не использовать local sampler
  как train runtime.
- [ ] Создать реальный `RLCluster` и проверить placement/sharing на declared mesh.
- [ ] Добавить output/logprob parity fixture между rollout model и train actor.

**Gate:** hardware-gated test создаёт `RLCluster`, загружает модель и делает
один rollout without update на target accelerator.

## 3. Agentic GRPO Training Loop

- [~] Добавить `run_agentic_grpo.py`: load profile -> evidence manifest -> task stream -> `RLCluster`
  -> Tunix Agentic `GRPOLearner` -> train/eval.
- [ ] Подключить `ToolAgent` и CrafText env factory к `GRPOLearner`, включая
  group key и `num_generations >= 2` для каждого исходного task.
- [ ] Определить terminal reward и metrics: environment return, success,
  invalid-action rate, episode length, tool-call distribution, KL and advantage.
- [ ] Добавить one-update integration: weights change, finite loss, valid group
  shape, no stale-policy samples after update.
- [ ] Добавить short fixed-task overfit/sanity run с deterministic stop condition.

**Gate:** CLI produces a trained actor checkpoint and an eval result that differs
from the frozen reference on the same fixed task set.

## 3.5. Agentic PPO / Critic Extension Lane

- [x] Добавить `tunix_craftext.agentic_ppo`: `AgenticPPOConfig`,
  `AgenticPPOTrainExample` и `AgenticPPOLearner` поверх upstream Tunix
  `AgenticRLLearner`, не через обычный text-only `PPOLearner`.
- [x] Подключить registered Tunix `ppo` actor loss и `ppo` value loss к
  `actor_trainer` и `critic_trainer`; отсутствие critic в `RLCluster` — hard error.
- [x] Преобразовать agentic trajectory в PPO train example с rollout/actor
  logprobs, reference logprobs, critic values, GAE advantages, returns и
  policy version.
- [ ] Добавить hardware-gated one-update test: actor и critic weights меняются,
  loss конечен, `update_actor()` и `update_critic()` вызываются одним base loop.
- [ ] Расширить этот lane до PPO-Lag/CPO: cost critic, cost returns,
  lagrange multiplier/projection objective и checkpoint metadata.

**Gate:** один Agentic PPO update на target accelerator проходит через тот же
`ToolAgent + CrafTextAgenticEnvironment + TrajectoryCollectEngine` transport и
создаёт checkpoint actor/reference/reward critic.

## 4. Evidence, Resume And Evaluation

- [ ] Store versioned YAML, git revision, model revision, topology, seed and
  package versions beside every run.
- [x] Добавить первый observability contract: `MetricRecord`,
  `ValidationTrajectoryRecord` и `JsonlRunLogger` пишут train/val/eval scalar
  metrics и ссылки на полные validation trajectories в versioned JSONL.
- [x] Добавить artifact sink contract и Comet ML adapter: `RunArtifact`,
  `ArtifactSink` и `CometMlSink` зеркалируют metrics, checkpoints, validation
  trajectories и visualization artifacts во внешний experiment tracker.
- [x] Добавить generic team logger adapter: `MappedLoggerSink` и
  `LoggerMethodMapping` позволяют подключить локальный logger команды через
  method mapping или direct callables без изменения core observability records.
- [x] Добавить standard artifact factories для checkpoint, weights,
  optimizer-state, train/val trajectories и validation visualizations, чтобы
  GRPO/PPO loop логировал все training artifacts единообразно.
- [ ] Подключить observability writer к реальному GRPO/PPO train loop: каждый
  update пишет loss/KL/return/success/invalid-action, checkpoint path и policy version.
- [ ] Export full validation trajectories for fixed task list; add TensorBoard
  only after JSONL schema is stable.
- [~] Add checkpoint save/restore for RLCluster/learner state: profile
  `evidence.checkpoints` now reaches Tunix `RLTrainingConfig`, which creates
  role-specific actor/critic checkpoint roots; hardware restart/preemption test
  proving the next update matches a continuous run remains open.
- [ ] Add deterministic evaluation command with fixed task list, success/reward
  metrics and reference-policy comparison.
- [ ] Record accelerator benchmark: compile/warmup, rollout tokens/s, env steps/s,
  update time, peak memory and degradation against a declared baseline.

**Gate:** rerunning the same profile produces comparable evidence and resumes
without silently changing model, task stream or topology.

## 5. CI And Release

- [ ] CPU CI: lint, unit, fake-agentic trajectory and config/schema tests on every PR.
- [ ] Accelerator CI: real Qwen `RLCluster` smoke, one GRPO update and checkpoint
  resume on an explicitly labelled hardware runner.
- [ ] Nightly performance lane with threshold/degradation report; do not infer
  scale-up from macOS results.
- [ ] Publish reproducibility card, known limitations, model licence record and
  migration guide from the old standalone PPO experiments.

## Deferred Work

These items are explicitly deferred until the golden pipeline passes all gates:

- custom token-level PPO/value head and Flashbax collector-to-PPO loop;
- SFT warm start, QLoRA/Qwix, state-dict model interop extensions;
- GRPO variants such as GSPO/DAPO beyond upstream Agentic GRPO;
- async multi-host MPI transport, multi-device scale thresholds and full release matrix.

## Current Baseline

Completed evidence includes CrafText adapters, deterministic JAX collectors,
Qwen sampler smoke, batched host-side rollout/replay, token contracts, Tunix
topology/config validation, and a fake-model `TrajectoryCollectEngine` multi-turn
test. None of these substitutes for the golden Agentic GRPO train workload.

The environment boundary is now explicitly layered: bare `CraftaxAdapter` owns
only JAX transition normalization; `CrafTextAdapter` adds a vendored instruction
wrapper, `TextEnvState` and world-preset context; `CagedCrafTextAdapter` adds the
text constraint aligned with that instruction. MegaPrompts receives the inner
Craftax `EnvState`, selected instruction, safety constraint and `world_preset` as
typed context. This is verified with real CrafText/Caged reset-step integration
tests and is a prerequisite for the golden multi-turn environment gate.
