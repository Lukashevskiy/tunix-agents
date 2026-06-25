# Примеры и notebooks

Runnable notebooks находятся в каталоге репозитория `examples/notebooks/`.

```bash
pyenv exec python -m uv sync --extra examples --extra prompts --extra envs --extra tunix
pyenv exec python -m uv run jupyter lab examples/notebooks
```

Начинайте с контрактов rollout, затем adapter среды, конвертации модели/LoRA и
`04_megaprompts_environment_to_prompt.ipynb` для реальной границы vendored template.

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
update и visualisation. Компактный Flax actor/critic пересчитывает `new_logprobs` и values;
`full_token_ppo_update()` учится по всем generated tokens из `token_mask`.

`12_full_cycle_craftext_training.ipynb` — real-model notebook того же полного цикла: Qwen/Tunix
rollout → replay evidence → `TextTrajectoryBatch` → real `TunixCausalLmActor.score_tokens()` →
phase profile JSON → full-token PPO smoke update. Веса всё ещё должны лежать локально и явно;
notebook не скачивает snapshot и не включает offline/mock fallback backend.

`13_replay_visualization.ipynb` открывает сохранённый replay JSON, показывает summary шагов,
reward/action timeline, prompt/completion и observation image, если replay содержит renderable
array. Он использует те же `load_trajectory()`/`normalize_image()`, что и pygame viewer ниже.

Тот же путь доступен вне Jupyter и сохраняет как raw replay, так и summary metrics:

```bash
.venv/bin/python scripts/run_text_episode.py --horizon 1
```

Для сохранённого notebook replay также доступен интерактивный pygame viewer:

```bash
PYTHONPATH=src .venv/bin/python scripts/visualize_trajectory.py \
  --trajectory artifacts/trajectories/qwen-craftext-full-notebook/env-0.json
```
