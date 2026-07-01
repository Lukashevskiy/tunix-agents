# Примеры и notebooks

Runnable notebooks находятся в каталоге репозитория `examples/notebooks/`.

```bash
uv sync --extra examples --extra prompts --extra envs --extra tunix
uv run jupyter lab examples/notebooks
```

Начинайте с контрактов rollout, затем adapter среды, конвертации модели/LoRA и
`04_megaprompts_environment_to_prompt.ipynb` для реальной границы vendored template.
Все notebooks, которые поднимают настоящий CrafText/CagedCrafText runtime, берут текст задачи
через `CrafTextTaskSampler` из vendored scenario instructions. В non-agentic batched rollout
используется fixed task, соответствующий `config.environment.instruction_index`; в agentic
GRPO notebooks training batches содержат и `goal`, и `instruction_index`, чтобы модель видела
тот же scenario row, на который reset-ится среда.

`06_qwen_craftext_manual_episode.ipynb` — это полный проверяемый LLM smoke: он требует явный
локальный снимок Qwen и проходит reset среды, рендеринг реального vendored MegaPrompts
`base` из `EnvState`, sampling Tunix, строгий decode с видимым fallback, одно действие CrafText и
replay v3 persistence.

`07_qwen_craftext_full_trajectory.ipynb` продолжает этот пример уже через итоговый batched
transport: `collect_batched_text_rollout()` вызывает MegaPrompts и один ordered Tunix/Qwen
batch на decision, enforce-ит action mask/fallback перед `vmap(CrafText.step)` и сохраняет
по одному replay на environment row через `replays_from_batched_rollout()`. Веса по-прежнему
должны заранее и явно находиться в `artifacts/models/qwen25-05b-instruct`.

`08_parallel_craftext_pipeline.ipynb` демонстрирует JAX-native transport, который downstream
text notebooks используют после decode: батч сред через `jax.vmap`, горизонт через
`jax.lax.scan`, actions/rewards `[T, B]` и отдельно compile/steady-state timing. Notebook 09/11/12
добавляют к этому transport batched Tunix completions, replay export и token PPO smoke.

`09_batched_qwen_craftext_rollout.ipynb` показывает уже реализованный bridge: B×T Qwen/CrafText
rollout, per-row terminal reset и export replay на каждую среду. `10_replay_to_token_ppo.ipynb`
без повторного LLM inference преобразует этот replay в `TextTrajectoryBatch`, показывает masks,
returns и token PPO loss mechanics.

`11_end_to_end_batched_qwen_ppo.ipynb` объединяет все эти стадии в одном run: параллельные
среды и Qwen requests, full rollout/replay, token batch, full-token trainable actor-critic PPO
update и visualisation. Компактный Flax actor/critic остаётся учебным механическим контуром для
проверки shapes/update, а не production LLM actor.

`12_full_cycle_craftext_training.ipynb` — real-model Gemma/Tunix notebook без smoke learner:
Gemma rollout → replay evidence → `TextTrajectoryBatch` → separate actor token scoring →
separate `TunixValueCritic` values → `evaluate_separate_llm_actor_critic_ppo()` with checked
returns/advantages/loss → phase profile JSON. Веса должны лежать локально и явно в
`artifacts/models/gemma3-270m-it`; notebook не скачивает snapshot и не включает offline/mock
fallback backend.

`13_replay_visualization.ipynb` открывает сохранённый replay JSON, показывает summary шагов,
reward/action timeline, prompt/completion и observation image, если replay содержит renderable
array. Он использует те же `load_trajectory()`/`normalize_image()`, что и pygame viewer ниже.

`14_generation_benchmark.ipynb` отдельно меряет Gemma generation pipeline: prompt/render →
batched Tunix generation → strict decode/fallback → CrafText step. Он сохраняет
`artifacts/benchmarks/gemma-generation-notebook.json` и profile JSON для сравнения batch size,
horizon и repeats.

`15_agentic_grpo_full_trainer.ipynb` — ручной operator notebook для critic-free GRPO path:
он по ячейкам поднимает profile/config, CrafText runtime, `CrafTextAgenticEnvironment`, prompt/tool
boundary, topology/mesh preflight, Qwen snapshot assets, `RLCluster`, `ToolAgent` и сам
`GRPOLearner`. Локальный scripted smoke остаётся как быстрая проверка grouped rewards без весов,
а настоящий trainer запускается отдельными heavy ячейками после readiness validation. Он нужен как рабочая проверка
agentic rollout/grouped rewards до добавления PPO critic/cost critic.

`16_server_grpo_object_training.ipynb` — более строгий серверный object-first notebook для
запуска на целевой GPU-машине. Он не вызывает `scripts/*.py`: вручную создаёт `JsonlRunLogger`,
пишет provenance/metrics/artifacts, собирает scripted validation trajectory, проверяет GPU/snapshot,
затем создаёт Qwen actor/reference assets, `RLCluster`, `GRPOLearner`, training batches из
CrafText scenario instructions через `CrafTextTaskSampler` и вызывает `learner.train(...)`
напрямую. После run последние ячейки читают `metrics.jsonl`,
`validation_trajectories.jsonl` и `artifacts.jsonl`, чтобы сразу проверить, что обучение,
validation и checkpoint evidence не потерялись.

`17_sync_vllm_craftext_rollout.ipynb` — первый реальный vLLM rollout notebook. Он берёт
backend profile и Tunix rollout knobs из `configs/inference/vllm/qwen25_05b_sync.yaml`, тот же
CrafText task из `CrafTextTaskSampler`, рендерит MegaPrompts, применяет Qwen chat
template через локальный tokenizer, вызывает `VllmInferenceEngine` через
`collect_generation_results_sync()` как contract smoke, затем синхронно через
`RequestsLlmBackend` использует существующий `collect_batched_text_rollout()` и сохраняет
replay artifacts. Это минимальный runnable sync vLLM путь без GRPO/PPO update.

`18_async_vllm_craftext_rollout.ipynb` использует тот же `GenerationBatch/GenerationResult`
контракт и `configs/inference/vllm/qwen25_05b_async.yaml`, но вызывает vLLM через
`collect_generation_results(..., max_in_flight=...)`.
Каждая строка environment batch становится отдельным async request, ответы собираются с
сохранением порядка, после чего один batched JAX `env.step` обновляет среды и replay
пишется по env row. Это полный single-node async rollout skeleton без Ray.

`19_host_prompt_threading_profile.ipynb` — диагностическая тетрадка для случая, когда vLLM
быстро обрабатывает batch, но sync rollout всё равно медленный и GPU простаивает. Она запускает
scripted backend без реальной модели, делает explicit warmup для JAX/XLA env compile, затем
несколько раз чередует serial prompt render и `HostBatchPolicy(prompt_workers=4)`, печатает
warmed median/min `phase_totals_ms()` и сохраняет
`artifacts/benchmarks/host-prompt-threading-latest.json`. Используйте её перед tuning vLLM,
чтобы отделить MegaPrompts/chat-template/Python host preparation от холодной XLA-компиляции.

`20_env_device_policy_benchmark.ipynb` — проверяет самую дорогую часть rollout после prompt
diagnostics: JAX env reset/step. Notebook должен начинаться с выбора `JAX_PLATFORMS` до импорта
JAX, сначала делает action terminal sweep, чтобы не спутать обычный rollout с reset storm, затем
сравнивает `vmap_only`, `jit(vmap)` на default backend и explicit
`EnvironmentDevicePolicy(backend=jax.default_backend())`, а также recommended CPU sidecar policy
через `cpu_environment_device_policy()`. Этот helper разворачивается в
`EnvironmentDevicePolicy(backend="cpu", jit_reset=True, jit_step=True, place_inputs=True,
snapshot_prompt_inputs=True)`. Для CPU env рядом с vLLM GPU запускайте
kernel с `JAX_PLATFORMS=cpu` до импорта JAX; для GPU env перезапустите kernel с CUDA-enabled JAX и сравните
`artifacts/benchmarks/env-device-policy-latest.json`.

`21_sync_vllm_grpo_learning.ipynb` — полный server runbook для Agentic GRPO learning:
profile/config/logger/provenance, sync vLLM smoke, pre/post validation trajectory evidence,
Tunix topology preflight, Qwen actor/reference loading, `RLCluster` с vLLM sync server-mode
rollout contract, `GRPOLearner.train(...)`, checkpoint artifact reference и JSONL метрики.
В отличие от smoke notebooks, эта тетрадка реально обновляет actor weights и требует локальный
Qwen snapshot, CUDA-ready runtime, Tunix и vLLM.

`22_external_vllm_sync_grpo_rollout.ipynb` — практический стартовый GRPO lane без Tunix
`RLCluster`: он повторяет direct vLLM path из notebook 17, собирает `B = group_size × group_count`
CrafText trajectories, сохраняет replay artifacts, затем строит `ExternalGrpoBatch` с
group-normalized advantages и summary metrics. Этот путь нужен, когда direct vLLM работает, но
Tunix internal JAX→vLLM weight-sync ещё блокируется несовместимостью worker RPC hooks. Он не
обновляет actor weights сам; его output — строгий evidence contract для следующего updater
слоя, LoRA/Qwix и будущего PPO critic.

Placement rule из rollout profiling остаётся важным: для standalone batched rollout рядом с
vLLM держите environment lane на CPU через `cpu_environment_device_policy()`, а
MegaPrompts/request preparation ускоряйте `HostBatchPolicy(prompt_workers=...)`. В Agentic
GRPO notebook не выставляйте `JAX_PLATFORMS=cpu` внутри kernel, потому что Tunix actor/reference
и role meshes должны видеть accelerator; CPU env/device-policy эксперименты запускаются
отдельным процессом/kernel до импорта JAX.

Тот же путь доступен вне Jupyter и сохраняет как raw replay, так и summary metrics:

```bash
uv run python scripts/run_text_episode.py --horizon 1
```

Для ручного управления агентом без LLM используйте manual controller. Он показывает текущую
CrafText instruction, text constraint, vitals/inventory, tactical ASCII map вокруг игрока и
legal actions, принимает action id/label из stdin и сохраняет каждый шаг в обычный replay
artifact:

```bash
uv run python scripts/manual_craftext_agent.py \
  --config configs/env/manual/caged_wood_achievements_energy.yaml \
  --horizon 16 \
  --replay-output artifacts/trajectories/manual-craftext-latest.json
```

Для сохранённого notebook replay также доступен интерактивный pygame viewer:

```bash
uv run python scripts/visualize_trajectory.py \
  --trajectory artifacts/trajectories/manual-craftext-latest.json
```

А для отчётов, сайта и Comet/локальных artifact sinks можно экспортировать GIF без открытия окна:

```bash
uv run python scripts/export_trajectory_gif.py \
  --trajectory artifacts/trajectories/manual-craftext-latest.json \
  --output artifacts/visualizations/manual-craftext-latest.gif \
  --fps 4 \
  --scale 4
```
