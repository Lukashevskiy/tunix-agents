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

`07_qwen_craftext_full_trajectory.ipynb` продолжает этот пример до полного короткого episode:
он вызывает `collect_text_episode()`, сохраняет replay с observation/action-mask/token provenance,
рисует rewards и actions, а затем выводит ленту CrafText observation frames. Веса по-прежнему
должны заранее и явно находиться в `artifacts/models/qwen25-05b-instruct`.

`08_parallel_craftext_pipeline.ipynb` демонстрирует текущую JAX-native parallel boundary:
батч сред через `jax.vmap`, горизонт через `jax.lax.scan`, actions/rewards `[T, B]` и отдельно
compile/steady-state timing. Он намеренно не выдаёт это за parallel Qwen inference: текущий
Qwen backend single-request, а batched actor/rollout service относится к будущему RLCluster этапу.

Тот же путь доступен вне Jupyter и сохраняет как raw replay, так и summary metrics:

```bash
.venv/bin/python scripts/run_text_episode.py --horizon 1
```

Для сохранённого notebook replay также доступен интерактивный pygame viewer:

```bash
PYTHONPATH=src .venv/bin/python scripts/visualize_trajectory.py \
  --trajectory artifacts/trajectories/qwen-craftext-full-notebook.json
```
