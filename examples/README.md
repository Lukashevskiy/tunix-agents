# Examples

Notebooks are deterministic, small and focused on one public contract.

```bash
.venv/bin/python -m pip install -e '.[examples]'
.venv/bin/python -m jupyter lab examples/notebooks
```

| Notebook | What it proves |
| --- | --- |
| `01_rollout_contract.ipynb` | Collect a `[T, B, ...]` trajectory. |
| `02_craftext_adapter.ipynb` | Normalize CrafText reset/step. |
| `03_model_interop_lora.ipynb` | Convert state dict and merge LoRA. |

Keep examples self-contained: fix seeds, avoid private data and do not import from `tests/`.
