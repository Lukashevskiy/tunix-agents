# Интеграция с Tunix

Tunix намеренно является необязательной зависимостью. Проект фиксирует официальный
[`google/tunix`](https://github.com/google/tunix) revision в
`compatibility/tunix.yaml`; он не должен использовать
непохожий PyPI-дистрибутив `tunix==0.0.0`.

Установите bridge-окружение явно:

```bash
uv sync --all-extras
```

Публичная граница намеренно узкая: Agentic `GRPOLearner`, `RLCluster` и
явные Qwen/Gemma loader/sampler boundaries в `tunix_adapter.py`. Локальный снимок Qwen 2.5
0.5B может sample-иться через `QwenTunixBackend`; локальный Gemma3 270M snapshot — через
`GemmaTunixBackend`. Наши контракты environment/prompt/replay остаются framework-neutral.

Перед каждым вызовом Qwen backend применяет заявленный tokenizer chat template
и вычисляет статическую ёмкость кеша, требуемую для power-of-two padding prompt в Tunix.
Он возвращает raw completion, latency, сгенерированные token ids и logprobs по токенам.
Слишком маленький кеш приводит к проектной `ValueError` до входа в Tunix.
Опциональный интеграционный smoke также подтверждает полный путь
`Qwen → decode/fallback → CrafText → replay v3` с локальными весами.

`text_trajectory_from_replay()` преобразует этот replay в упакованный JAX
`TextTrajectoryBatch`: необработанные prompt/generated token ids, behaviour logprobs,
маски token/policy и reward среды на каждом итоговом токене completion.
Шаги с fallback остаются аудируемыми, но исключаются из `policy_mask`, чтобы
будущее PPO/SFT обновление не могло silent learn из не-модельного действия.

`masked_token_returns()` и `masked_token_ppo_loss()` соответствуют чистому JAX loss
boundary: они работают только с корректными `[B, T]` токенами и игнорируют padding/
fallback mask позиции. Для production-shaped LLM path `TunixCausalLmActor.score_actor_tokens()`
пересчитывает actor `new_logprobs` и entropy, `TunixValueCritic.score_values()` отдельно считает
critic values, а `evaluate_separate_llm_actor_critic_ppo()` объединяет эти роли только на
objective boundary. Компактный `PromptConditionedTokenActorCritic` остаётся unit/notebook smoke
для проверки mechanics update, но notebook 12 больше не использует его как финальный контур.

Для production критика `QwenTunixBackend.hidden_states()` и `GemmaTunixBackend.hidden_states()`
открывают финальные `[B, T, D]` признаки закреплённой модели через `skip_lm_head=True`.
Прикрепление, обучение и checkpointing value head остаются отдельным явным workload шагом;
текущий Gemma notebook проверяет значения critic forward/evaluation, но ещё не делает trainable
LoRA/Qwix update.

Базовая среда не импортирует Tunix. Это сохраняет `make test`, сбор CrafText и smoke
обучение Flax/Optax независимыми от тяжёлого стека модели и tokenizer, сохраняя при этом
точный и воспроизводимый bridge source.

## LLM actor backbone boundary

Нормальный actor теперь задаётся не compact Flax bridge, а интерфейсом `LlmActor`.
Он имеет две обязательные операции: `generate_batch()` для взаимодействия со средой и
`score_tokens()` для пересчёта token logprobs, critic values и entropy на уже собранном
`TextTrajectoryBatch`. Это форма, в которую должны лечь production Qwen/Gemma actors.

`configs/models/gemma3_270m_instruction.yaml` выбран как маленький Gemma3 270M instruction
backbone candidate: установленный Tunix содержит `gemma3_270m_it`, Gemma chat parser и Gemma3
model modules. Profile loader проверяет этот YAML без загрузки весов. Практика из
`chuyishang/jax-lm` применяется как engineering discipline — uv/test-first/static-shape
separation of model/tests/scripts — но сам `jax-lm` не становится runtime dependency.

`DeterministicLlmActor` остаётся TDD double, но production-shaped слой уже вынесен в
`tunix_actor.py`: `TunixCausalLmActor` соединяет ordered `BatchLlmBackend`, causal LM модель и
малый `LinearValueHead`. `generate_batch()` идёт через sampler/backend, `score_actor_tokens()`
собирает actor logprobs с авторегрессивных позиций и entropy, а `critic()` возвращает отдельный
`TunixValueCritic` для values. `QwenTunixActor` и `GemmaTunixActor` имеют factories от явного
локального snapshot; никакой builder не скачивает веса как побочный эффект unit path или импорта.

Ни одни веса модели не загружаются как побочный эффект установки или обычного теста.
Опциональный реальный Qwen smoke запускается только тогда, когда явный локальный снимок
существует в `artifacts/models/qwen25-05b-instruct`.

Tunix владеет распределённым исполнением, но проект должен объявлять топологию:
`RLCluster` получает versioned `role_to_mesh` mapping для actor, rollout, reference
и PPO critic. Tunix может применять sharding/offload; проект не добавляет в этом репозитории
второй GPU scheduler. Решение архитектуры и отличие от локального sampler задокументированы в
[ADR 0004](adr/0004-tunix-cluster-topology.md).

Топология живёт в `configs/topology/`: `qwen_local_smoke.yaml` размещает все роли на
устройстве 0, а `qwen_four_device_colocated.yaml` документирует четырехустройственный mesh.
Оба профиля строго валидируются до построения accelerator workload; профиль, запрашивающий
недоступные устройства, завершается раньше. `tunix_role_to_meshes()` — единственный адаптер,
сопоставляющий эти именованные роли с официальным Tunix `Role` enum.

Для Agentic GRPO используется отдельный `qwen_agentic_grpo_local.yaml`: он намеренно
содержит только `actor`, `rollout` и `reference`. `AgenticGrpoWorkloadSpec` строит
соответствующий `ClusterConfig` с recomputed logprobs; critic остаётся обязательным
только для будущего PPO profile.

`load_agentic_grpo_qwen_assets()` создаёт независимые copies из одного explicit
local safetensors snapshot: trainable actor хранится в `float32`, а frozen reference
в `bfloat16`. `build_agentic_grpo_cluster()` получает уже созданные assets и вызывает
публичный `RLCluster`; этим загрузка весов отделена от cluster construction и остаётся
hardware-gated.

На одном GPU с 12GB VRAM это следует считать smoke/bring-up режимом, а не обещанной
training-конфигурацией. Qwen 0.5B GRPO держит actor + rollout/reference роли на одном
устройстве и может упереться в KV-cache, optimizer state и temporary logits; поэтому
профиль должен начинаться с маленьких `max_new_tokens`, `kv_cache_size`,
`mini_batch_size=2`, `train_micro_batch_size=2`, `rollout_micro_batch_size=1`.
Agentic PPO тяжелее GRPO, потому что добавляет critic model/trainer; для 12GB первым
реалистичным sanity path является Gemma 270M или LoRA/Qwix-only update, а не full
Qwen actor+critic в fp32.

Для PPO path добавлены `load_ppo_qwen_assets()`, `load_ppo_gemma_assets()` и
`build_ppo_cluster()`. Они создают actor/reference/tokenizer и обязательный critic model до
вызова public `RLCluster(actor=..., critic=..., reference=..., tokenizer=...)`. Qwen critic
использует upstream Tunix `create_critic_model()`, который заменяет `lm_head`. Gemma3 не имеет
`lm_head` — logits считаются через `embedder.decode` — поэтому `create_value_critic_from_actor()`
делает NNX-copy actor, добавляет scalar `value_head` и подменяет `compute_final_logits()` на
value-output. Это временный совместимый bridge до upstream Gemma critic factory; heavy model
loading остаётся hardware-gated и не происходит в unit/docs path.

`scripts/run_agentic_grpo.py` связывает task batches, `ToolAgent`,
`CrafTextAgenticEnvironment`, Qwen assets, `RLCluster` и upstream
`GRPOLearner`. На локальном CPU он завершится до загрузки весов: pinned Tunix
Qwen vanilla sampler пока не поддерживает его singleton sharded-gather path.
Реальный one-update smoke предназначен для accelerator runner; флаг
`--allow-cpu-smoke` существует только для воспроизведения upstream failure.

После аудита внешних практик `jax-lm` и NVIDIA JAX-Toolbox это ограничение оформлено как
архитектурная граница, а не как временный notebook-костыль. `jax-lm` разделяет semantic
sharding modes и не считает symbolic `tp` degree 1 полноценным tensor-parallel режимом.
NVIDIA JAX-Toolbox отделяет rollout generation от trainer mesh через inference/offload
boundary. Поэтому `validate_agentic_grpo_preflight()` теперь отклоняет Qwen +
`vanilla-jax-sharded` + `fsdp,tp` до загрузки весов; evidence/scripted checks остаются
разрешены, а real train ждёт `single-device-jax`, `vllm-offload` или upstream Tunix
embedding-gather fix. Решение зафиксировано в
[ADR 0006](adr/0006-rollout-generation-boundary.md).

Перед real weights path есть два локальных gate. `--dry-run` валидирует GRPO profile,
topology, static preflight и evidence manifest без model allocation. `--scripted-smoke`
использует тот же `CrafTextAgenticEnvironment`, `ToolAgent` и Tunix `TrajectoryCollectEngine`,
но подставляет scripted tool calls вместо LLM. Он собирает несколько generations одного task
group, считает суммарные rewards и GRPO-style group-normalized advantages. Это проверяет
agentic rollout/tool/reward/grouping semantics до critic-free `GRPOLearner` запуска; будущие
PPO-Lag/CPO слои должны переиспользовать этот transport и добавить value/cost critic, а не
переписывать CrafText environment loop.

Для critic-backed agentic path добавлен проектный bridge `tunix_craftext.training.agentic_ppo`.
Он не использует обычный text-only `PPOLearner`: вместо этого наследуется от upstream
Tunix `AgenticRLLearner`, то есть сохраняет тот же async `RolloutOrchestrator`,
`TrajectoryCollectEngine`, `ToolAgent` и `BaseTaskEnv` transport. `AgenticPPOLearner`
подключает registered Tunix `ppo` actor loss и `ppo` value loss, требует critic model
в `RLCluster`, а `_process_results()` превращает agentic trajectory в PPO-shaped
`AgenticPPOTrainExample` с `old_per_token_logps`, critic `old_values`, GAE
`advantages/returns` и `policy_version`. Базовый Tunix loop после этого сам вызывает
`update_actor()` и `update_critic()`. Следующий hardware-gated gate — один реальный
Agentic PPO update на Qwen/Gemma actor/reference/critic assets.

`tests/integration/test_tunix_topology_hardware.py` намеренно hardware-gated: он пропускает
тест на системах с менее чем четырьмя видимыми устройствами и проверяет четырехустройственный
профиль на accelerator runner. Это проверяет declaration placement, а не масштабную производительность;
последняя относится к performance lane.

## Sync-first training path

Ближайший MVP следует [Anakin JAX-first pattern](https://arxiv.org/abs/2104.06272): после
host-side prompt/rendering фиксированной формы `TextTrajectoryBatch` попадает в
`FlashbaxTextReplay`. Его `init → add → sample` реально проходят под `jax.jit`; Flashbax
хранит bounded staging window, а не превращает PPO в off-policy алгоритм. До следующего
policy update из него можно семплировать только current-window behaviour data.

Установите зависимости данного пути так:

```bash
uv sync --extra tunix --extra replay --extra lora --extra loop
```

Qwix — принятый путь для будущего LoRA/QLoRA. Его включение в train workload будет возможно
после architecture-specific output-parity, gradient и checkpoint metadata fixtures; Qwix не
подменяет существующий безопасный state-dict LoRA merge. CommonLoopUtils зарезервирован для
structured metrics/checkpoints на границе update, а mpi4jax — только для будущей multi-host
async фазы после проверки MPI toolchain. Состояния всех четырёх интеграций зафиксированы в
`compatibility/training-stack.yaml` и [ADR 0005](adr/0005-sync-first-training-stack.md).

## Batched rollout boundary

`QwenTunixBackend.complete_batch()` принимает ordered batch одинаковых `LlmRequest` и передаёт
весь список chat prompts в **один** публичный Tunix sampler invocation. Ответы сохраняют
cardinality и порядок input batch, а также индивидуальные generated/prompt token ids и logprobs.
Параметры `max_new_tokens` и `temperature` должны совпадать: это static sampler contract, а не
повод скрытно разбить запрос на последовательные вызовы. `latency_ms` в каждой строке — wall
time общего batch вызова, его нельзя суммировать.

Реальный fixture закреплённого Qwen 0.5B подтверждает batch size 2. Это bridge к
`RLCluster` `ROLLOUT` role, но ещё не сам distributed RL workload: для последнего нужны
trainable actor/critic/reference, `RLTrainingConfig`, `RolloutConfig` и hardware-gated cluster
fixture. После этого batch completions будут декодироваться в `[B]` action ids и подаваться в
`jax.vmap(CrafText.step)`.

`collect_batched_text_decision()` уже реализует один synchronous decision transport: host-side
batch `EnvState → MegaPrompts → complete_batch → strict decode/fallback`, после чего выполняет
один `jax.vmap(CrafTextAdapter.step)` по `[B]` action ids. Real Qwen+CrafText fixture подтверждает
batch size 2. Multi-turn semantics (`terminated | truncated → reset only that row`) вынесены в
`collect_batched_text_rollout()`, который является sync precursor будущего consumer
`RLCluster.ROLLOUT`, когда будут созданы trainable роли actor/critic/reference.

## Multi-turn Agentic CrafText

Профиль `tunix` использует закреплённый Tunix 0.1.7 и его Agentic RL API для
многоходового взаимодействия. `build_craftext_tool_agent()` создаёт Qwen
`ToolAgent` с единственным вызовом `craftext_step(action=...)`.
`CrafTextAgenticEnvironment` - module-level `BaseTaskEnv`, который создаётся из
serializable task (`goal`, `seed`, optional `horizon`) и `config_path`; это
позволяет Tunix worker-ам создавать независимые GRPO generations без передачи
JAX state или замыканий между потоками.

После хода среда валидирует action catalogue и current action mask, затем возвращает
следующий MegaPrompts prompt как `tool_outputs`,
поэтому `TrajectoryCollectEngine` Tunix ведёт историю диалога, считает
многоходовую траекторию и сохраняет token-level данные. Невалидный tool call
остаётся наблюдаемым ответом среды с нулевой наградой; он не превращается в
скрытый fallback action.

Этот слой совместим с Agentic GRPO `RLCluster`, но не создаёт модели автоматически:
actor/rollout/reference и accelerator mesh остаются явной конфигурацией
production workload.

`collect_batched_text_rollout()` расширяет transport до fixed horizon `[T, B]`: после каждого
шага сохраняется terminal transition, а `terminated | truncated` rows получают reset только для
следующего шага (другие states/dialogs продолжаются). `replays_from_batched_rollout()` создаёт
по одному replay v3 на environment row; каждый из них напрямую совместим с
`text_trajectory_from_replay()` и masked token PPO learning contract.

Перед каждым `CrafTextAdapter.step` итоговый text rollout сверяет decoded action с текущим
`action_mask`. Если модель выбирает masked action, это не попадает в среду молча: transport
либо падает в `invalid_action="error"` режиме, либо использует явный fallback и сохраняет
`masked_action` вместе с `fallback_used` в replay evidence. Поэтому будущий learner может
исключать такие токены из `policy_mask`, а audit может восстановить причину fallback.

Соответствующие versioned profiles — `configs/models/gemma3_270m_instruction.yaml`
и `configs/models/qwen25_05b_instruction.yaml`. Их зафиксированные флаги download/license
намеренно остаются `false`: они описывают портируемый репозиторий, а не приватное состояние
одной машины.
