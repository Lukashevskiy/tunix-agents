# Примеры и notebooks

Runnable notebooks находятся в каталоге репозитория `examples/notebooks/`.

```bash
pyenv exec python -m uv sync --extra examples --extra prompts
pyenv exec python -m uv run jupyter lab examples/notebooks
```

Начинайте с контрактов rollout, затем adapter среды, конвертации модели/LoRA и
`04_megaprompts_environment_to_prompt.ipynb` для реальной границы vendored template.

`06_qwen_craftext_manual_episode.ipynb` — это полный проверяемый LLM smoke: он требует явный
локальный снимок Qwen и проходит reset среды, рендеринг реального vendored MegaPrompts
`base` из `EnvState`, sampling Tunix, строгий decode с видимым fallback, одно действие CrafText и
replay v3 persistence.

Тот же путь доступен вне Jupyter и сохраняет как raw replay, так и summary metrics:

```bash
.venv/bin/python scripts/run_text_episode.py --horizon 1
```
