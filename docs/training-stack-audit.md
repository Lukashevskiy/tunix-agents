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
Он физически вынесен в `tunix_craftext.research`; старые top-level пути остались только как
thin compatibility shims. Этот слой нельзя использовать как acceptance gate для production
RLCluster обучения.

## Карта текущих слоёв

| Слой | Файлы | Статус | Что делать |
| --- | --- | --- | --- |
| Agentic environment transport | `env/agentic_craftext.py`, `env/prompts.py`, `env/text_policy.py`, `env/runtime.py`, `adapters/craftext.py` | production boundary | Оставить как общий транспорт для GRPO/PPO/PPO-Lag/CPO. |
| Tunix topology/workload | `tunix/topology.py`, `tunix/rlcluster_workload.py`, `tunix/preflight.py`, `grpo_profile.py` | production boundary | Уже вынесено в semantic package `tunix_craftext.tunix`; старые root modules остались thin compatibility shims. |
| Agentic GRPO | `training/agentic_grpo_smoke.py`, `scripts/run_agentic_grpo.py`, `configs/grpo/qwen_agentic_local.yaml` | текущий golden path до heavy update | Довести до real one-update на accelerator/local snapshot. |
| Agentic PPO | `training/agentic_ppo.py`, `tests/unit/test_agentic_ppo.py` | current extension lane | Добавить hardware-gated one-update actor+critic, затем cost critic для PPO-Lag/CPO. |
| Tunix LLM backend | `models/tunix_adapter.py`, `models/tunix_actor.py`, `models/llm_actor.py`, `models/profile.py` | production/research bridge | Следующий split внутри `models`: loaders, sampler backend, scoring и profiles. |
| Local PPO mechanics | `research/algorithms.py`, `research/algorithm_registry.py`, `research/learner.py`, `research/llm_ppo.py`, `checkpoints.py` | research/smoke | Уже вынесено из production namespace; старые top-level пути оставлены как thin compatibility shims. |
| Replay staging | `training/flashbax_replay.py`, `artifacts/replay.py`, `artifacts/text_trajectory.py` | reusable staging | Оставить, но документационно ограничить on-policy/bounded staging. |
| Local rollout examples | `rollouts/reference.py`, `rollouts/batched.py`, `rollouts/hybrid.py`, `rollouts/text_episode.py`, `rollouts/random_policy.py` | reference/contract boundary | `reference.py` оставить fixed-shape baseline; `batched.py` оставить synchronous host+JAX precursor; `hybrid.py` использовать как PPO-ready evidence contract с actor logprobs, critic values, token masks и step masks. Не считать это самостоятельным trainer. |
| Interop | `interop/*` | support module | Оставить отдельно; Qwix/LoRA integration делать здесь. |

## Как сейчас реализован GRPO

GRPO реализуется не через локальный PPO loss, а через upstream Tunix Agentic RL контур:

1. Profile `configs/grpo/qwen_agentic_local.yaml` задаёт модель, topology, workload и evidence paths.
2. `grpo_profile.py` валидирует profile, пишет hash/provenance/package evidence.
3. `tunix/topology.py` объявляет роли `actor`, `rollout`, `reference`; critic намеренно отсутствует.
4. `tunix/rlcluster_workload.py::build_agentic_grpo_cluster_config()` строит Tunix `ClusterConfig` с
   `RLTrainingConfig`, `RolloutConfig`, checkpoint root и recomputed logprobs.
5. `env/agentic_craftext.py` адаптирует CrafText в multi-turn tool-call environment для Tunix
   `TrajectoryCollectEngine`.
6. `scripts/run_agentic_grpo.py` собирает preflight/evidence, может выполнить `--dry-run` и
   `--scripted-smoke`, а heavy path должен создать real `RLCluster` и `GRPOLearner`.

Сильная сторона: GRPO не требует trainable critic и поэтому является реалистичным первым запуском
на одной потребительской GPU. Слабое место: пока нет hardware-gated теста, который доказывает, что
реальный actor checkpoint изменился после `GRPOLearner` update.

## Как сейчас реализован PPO

Есть два разных PPO-контура, и их нельзя смешивать:

### 1. Agentic PPO поверх Tunix

`training/agentic_ppo.py` добавляет critic-backed PPO subclass поверх upstream
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

`research/algorithms.py`, `research/learner.py`, `research/llm_ppo.py` и `checkpoints.py`
проверяют математику и формы: GAE, clipped PPO, full-token update, mask semantics,
local Flax/Optax/Orbax checkpoint restore.
Это ценно для TDD, но не заменяет Tunix Agentic PPO, потому что не владеет реальным
distributed actor/reference/critic lifecycle.

### 3. Hybrid PPO rollout contract

Внешний PPO-аудит отдельно указал, что `jax.lax.scan` с динамически растущей LLM-историей
приведёт к статическому padding/KV-cache overhead и не должен быть production collector.
Поэтому добавлен `rollouts/hybrid.py`: он фиксирует данные, которые обязан вернуть реальный
host-orchestrated rollout перед PPO update:

- discrete `action_ids` для `jax.vmap(CrafTextAdapter.step)`;
- `prompt_tokens` и `generation_tokens` как evidence/model inputs;
- `actor_log_probs` токенов rollout policy;
- `values` от critic role;
- `generation_token_mask` для padding ответа;
- `step_mask` для post-terminal padding батча;
- optional `action_mask`, который позже должен перейти в Tunix logits processor.

Этот слой согласует `batched_rollout.py`/notebooks с будущим `AgenticPPOLearner` input contract,
но не возвращает старый local PPO learner в production namespace.

## Что вырезано сейчас

Из индекса удалены Jupyter checkpoint-файлы:

- `examples/.ipynb_checkpoints/README-checkpoint.md`
- `examples/notebooks/.ipynb_checkpoints/05_caged_random_policy_trajectory-checkpoint.ipynb`
- `examples/notebooks/.ipynb_checkpoints/06_qwen_craftext_manual_episode-checkpoint.ipynb`

Они являются локальными редакторскими артефактами, не документацией и не evidence. `.gitignore`
теперь запрещает повторно добавлять `.ipynb_checkpoints/`.

## Что не стоит вырезать прямо сейчас

- `research/learner.py`, `research/algorithms.py`, `research/llm_ppo.py`: нужны notebook
  10/11/12 и unit tests для loss contracts. Они уже не production modules; старые
  `tunix_craftext.research.learner`, `tunix_craftext.research.algorithms` и `tunix_craftext.research.llm_ppo`
  остаются только для совместимости.
- `rollouts/reference.py`, `rollouts/batched.py`, `rollouts/text_episode.py`: нужны reference/perf/env evidence.
- `training/flashbax_replay.py`: нужен для будущего bounded staging и синхронного PPO window.
- `artifacts/checkpoints.py`: локальный Orbax smoke остаётся regression test для optimizer-state restore,
  даже если production Tunix checkpoints принадлежат Tunix trainers.

## Рекомендуемая физическая реорганизация

Делать отдельными маленькими коммитами:

1. Создан первый semantic package `src/tunix_craftext/tunix/`:
   `topology.py`, `rlcluster_workload.py`, `preflight.py`.
2. `src/tunix_craftext/research/` уже содержит:
   `algorithms.py`, `learner.py`, `llm_ppo.py`, `algorithm_registry.py`.
3. Compatibility imports в старых путях оставить на один-два цикла:
   `from tunix_craftext.research.learner import ...`.
4. Созданы semantic packages `core/`, `env/`, `rollouts/`, `models/`, `training/`,
   `artifacts/`; старые top-level пути являются compatibility shims.
5. Следующим отдельным коммитом разбить `models/tunix_adapter.py`:
   `loaders.py`, `samplers.py`, `scoring.py`, `profiles.py`.
6. Добавить migration test, который запрещает production modules импортировать
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
