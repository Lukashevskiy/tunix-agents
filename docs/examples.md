# Examples and notebooks

Runnable notebooks live in the repository directory `examples/notebooks/`.

```bash
pyenv exec python -m uv sync --extra examples --extra prompts
pyenv exec python -m uv run jupyter lab examples/notebooks
```

Start with rollout contracts, then the environment adapter, model conversion/LoRA, and
`04_megaprompts_environment_to_prompt.ipynb` for the real vendored template boundary.
