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
| `06_qwen_craftext_manual_episode.ipynb` | Run reset → Qwen prompt → decode/fallback → CrafText step → replay. |
| `07_qwen_craftext_full_trajectory.ipynb` | Collect a full Qwen-driven CrafText replay and visualize actions, rewards and observation frames. |
| `08_parallel_craftext_pipeline.ipynb` | Run parallel CrafText environments with `vmap` and a compiled horizon with `lax.scan`. |
| `09_batched_qwen_craftext_rollout.ipynb` | Collect B×T Qwen/CrafText rollout, terminal resets and per-environment replays. |
| `10_replay_to_token_ppo.ipynb` | Convert replay evidence to token batches, masks, returns and PPO loss inputs. |

Keep examples self-contained: fix seeds, avoid private data and do not import from `tests/`.
