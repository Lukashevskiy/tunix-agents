# Аудит training stack: PPO, GRPO и границы модулей

Этот документ фиксирует, что в проекте сейчас является production path, что остаётся
исследовательским слоем, а что нужно вырезать или перенести. Главная цель — не смешивать
локальные PPO-smoke механики с реальным Tunix `RLCluster`/Agentic learner контуром.

## Короткий вывод

Production-направление проекта — **Tunix Agentic GRPO first**, затем **Agentic PPO extension**.
Оно проходит через `RLCluster`, публичные Tunix trainers, `CrafTextAgenticEnvironment`,
`ToolAgent`, profile/evidence/checkpoint contracts и future hardware-gated one-update tests.

Локальный `algorithms.py` / `learner.py` / `llm_ppo.py` стек полезен как TDD-слой для форм,
масок, GAE, token-level losses и notebook education, но это больше не основной trainer.
Его нужно маркировать как `research/smoke` и не использовать как acceptance gate для
production RLCluster обучения.

## Карта текущих слоёв

| Слой | Файлы | Статус | Что делать |
| --- | --- | --- | --- |
| Agentic environment transport | `agentic_craftext.py`, `prompts.py`, `text_policy.py`, `runtime.py`, `adapters/craftext.py` | production boundary | Оставить как общий транспорт для GRPO/PPO/PPO-Lag/CPO. |
| Tunix topology/workload | `tunix_topology.py`, `rlcluster_workload.py`, `preflight.py`, `grpo_profile.py` | production boundary | Оставить; позднее split на `training/topology.py`, `training/workload.py`, `training/assets.py`. |
| Agentic GRPO | `agentic_grpo_smoke.py`, `scripts/run_agentic_grpo.py`, `configs/grpo/qwen_agentic_local.yaml` | текущий golden path до heavy update | Довести до real one-update на accelerator/local snapshot. |
| Agentic PPO | `agentic_ppo.py`, `tests/unit/test_agentic_ppo.py` | current extension lane | Добавить hardware-gated one-update actor+critic, затем cost critic для PPO-Lag/CPO. |
| Tunix LLM backend | `tunix_adapter.py`, `tunix_actor.py`, `llm_actor.py`, `model_profile.py` | смешанный production/research bridge | Разбить большой adapter на loaders, sampler backend, scoring и model profile modules. |
| Local PPO mechanics | `algorithms.py`, `algorithm_registry.py`, `learner.py`, `llm_ppo.py`, `checkpoints.py` | research/smoke | Перенести в `tunix_craftext/research/` или `tunix_craftext/smoke/` после migration imports. |
| Replay staging | `flashbax_replay.py`, `replay.py`, `text_trajectory.py` | reusable staging | Оставить, но документационно ограничить on-policy/bounded staging. |
| Local rollout examples | `rollout.py`, `batched_rollout.py`, `episode.py`, `random_policy.py` | reference/examples | Оставить для env/perf/contracts; не считать RLCluster trainer. |
| Interop | `interop/*` | support module | Оставить отдельно; Qwix/LoRA integration делать здесь. |

## Как сейчас реализован GRPO

GRPO реализуется не через локальный PPO loss, а через upstream Tunix Agentic RL контур:

1. Profile `configs/grpo/qwen_agentic_local.yaml` задаёт модель, topology, workload и evidence paths.
2. `grpo_profile.py` валидирует profile, пишет hash/provenance/package evidence.
3. `tunix_topology.py` объявляет роли `actor`, `rollout`, `reference`; critic намеренно отсутствует.
4. `rlcluster_workload.py::build_agentic_grpo_cluster_config()` строит Tunix `ClusterConfig` с
   `RLTrainingConfig`, `RolloutConfig`, checkpoint root и recomputed logprobs.
5. `agentic_craftext.py` адаптирует CrafText в multi-turn tool-call environment для Tunix
   `TrajectoryCollectEngine`.
6. `scripts/run_agentic_grpo.py` собирает preflight/evidence, может выполнить `--dry-run` и
   `--scripted-smoke`, а heavy path должен создать real `RLCluster` и `GRPOLearner`.

Сильная сторона: GRPO не требует trainable critic и поэтому является реалистичным первым запуском
на одной потребительской GPU. Слабое место: пока нет hardware-gated теста, который доказывает, что
реальный actor checkpoint изменился после `GRPOLearner` update.

## Как сейчас реализован PPO

Есть два разных PPO-контура, и их нельзя смешивать:

### 1. Agentic PPO поверх Tunix

`agentic_ppo.py` добавляет critic-backed PPO subclass поверх upstream
`tunix.rl.agentic.agentic_rl_learner.AgenticRLLearner`.

Что уже хорошо:

- `AgenticPPOConfig` явно требует `num_generations == 1`;
- отсутствие critic в `RLCluster` — hard error;
- actor trainer получает Tunix registered `ppo` policy loss;
- critic trainer получает Tunix registered `ppo` value loss;
- trajectory превращается в PPO train example с `old_per_token_logps`, reference logprobs,
  critic values, GAE advantages, returns и policy version;
- checkpoint root уже прокидывается в Tunix `RLTrainingConfig`.

Что ещё не доказано:

- real one-update actor+critic на accelerator/local snapshot;
- физическое создание `actor/` и `critic/` checkpoints;
- restart/resume parity после preemption;
- cost critic / constrained objective для PPO-Lag и CPO.

### 2. Локальный PPO research stack

`algorithms.py`, `learner.py`, `llm_ppo.py` и `checkpoints.py` проверяют математику и формы:
GAE, clipped PPO, full-token update, mask semantics, local Flax/Optax/Orbax checkpoint restore.
Это ценно для TDD, но не заменяет Tunix Agentic PPO, потому что не владеет реальным
distributed actor/reference/critic lifecycle.

## Что вырезано сейчас

Из индекса удалены Jupyter checkpoint-файлы:

- `examples/.ipynb_checkpoints/README-checkpoint.md`
- `examples/notebooks/.ipynb_checkpoints/05_caged_random_policy_trajectory-checkpoint.ipynb`
- `examples/notebooks/.ipynb_checkpoints/06_qwen_craftext_manual_episode-checkpoint.ipynb`

Они являются локальными редакторскими артефактами, не документацией и не evidence. `.gitignore`
теперь запрещает повторно добавлять `.ipynb_checkpoints/`.

## Что не стоит вырезать прямо сейчас

- `learner.py`, `algorithms.py`, `llm_ppo.py`: нужны notebook 10/11/12 и unit tests для loss
  contracts. Их лучше переносить только вместе с compatibility imports.
- `rollout.py`, `batched_rollout.py`, `episode.py`: нужны reference/perf/env evidence.
- `flashbax_replay.py`: нужен для будущего bounded staging и синхронного PPO window.
- `checkpoints.py`: локальный Orbax smoke остаётся regression test для optimizer-state restore,
  даже если production Tunix checkpoints принадлежат Tunix trainers.

## Рекомендуемая физическая реорганизация

Делать отдельными маленькими коммитами:

1. Создать `src/tunix_craftext/training/`:
   `agentic_grpo.py`, `agentic_ppo.py`, `workload.py`, `topology.py`, `preflight.py`.
2. Создать `src/tunix_craftext/research/`:
   `algorithms.py`, `learner.py`, `llm_ppo.py`, `algorithm_registry.py`.
3. Оставить compatibility imports в старых путях на один-два цикла:
   `from tunix_craftext.research.learner import ...`.
4. Разбить `tunix_adapter.py`:
   `tunix_loaders.py`, `tunix_sampler.py`, `tunix_scoring.py`, `model_profiles.py`.
5. Добавить migration test, который запрещает production modules импортировать
   `tunix_craftext.research.*`.

## Acceptance gates перед записью “готовы мерить обучение”

- `pytest tests/unit/test_agentic_ppo.py tests/unit/test_rlcluster_workload.py`
- `pytest tests/unit/test_grpo_profile.py tests/unit/test_run_agentic_grpo.py`
- `PYTHONPATH=src .venv/bin/python scripts/run_agentic_grpo.py --profile configs/grpo/qwen_agentic_local.yaml --dry-run`
- `PYTHONPATH=src .venv/bin/python scripts/run_agentic_grpo.py --profile configs/grpo/qwen_agentic_local.yaml --scripted-smoke`
- hardware/local snapshot gate: one Agentic GRPO update changes actor checkpoint;
- hardware/local snapshot gate: one Agentic PPO update changes actor and critic checkpoints;
- resume gate: next update after restore matches continuous run on fixed fixture.

## 12GB GPU expectation

На одной GPU с 12GB VRAM разумная цель — сначала GRPO smoke/update на маленьком backbone
(`Gemma 270M` или small Qwen) с LoRA/Qwix и короткими sequence limits. PPO тяжелее, потому что
держит trainable actor и critic плюс frozen reference/rollout semantics; полноценный Qwen actor
+ critic в fp32 на 12GB, скорее всего, будет memory-bound. Поэтому порядок остаётся:
GRPO → LoRA/Qwix → checkpoint/resume → PPO critic → cost critic/PPO-Lag/CPO.
