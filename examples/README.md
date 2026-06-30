# Examples

Notebooks are deterministic, small and focused on one public contract.

```bash
uv sync --extra examples --extra prompts --extra envs --extra tunix
uv run jupyter lab examples/notebooks
```

| Notebook | What it proves |
| --- | --- |
| `01_rollout_contract.ipynb` | Collect a `[T, B, ...]` trajectory. |
| `02_craftext_adapter.ipynb` | Normalize CrafText reset/step. |
| `03_model_interop_lora.ipynb` | Convert state dict and merge LoRA. |
| `06_qwen_craftext_manual_episode.ipynb` | Run reset → Qwen prompt → decode/fallback → CrafText step → replay. |
| `07_qwen_craftext_full_trajectory.ipynb` | Collect final batched Qwen/Tunix → MegaPrompts → CrafText replays and visualize one environment trace. |
| `08_parallel_craftext_pipeline.ipynb` | Run the JAX-parallel CrafText transport that downstream text rollout notebooks reuse before replay/token batching. |
| `09_batched_qwen_craftext_rollout.ipynb` | Collect B×T Qwen/CrafText rollout, terminal resets and per-environment replays. |
| `10_replay_to_token_ppo.ipynb` | Convert replay evidence to token batches, masks, returns and PPO loss inputs. |
| `11_end_to_end_batched_qwen_ppo.ipynb` | Run the full batched Env → MegaPrompts → Qwen → replay → token PPO loss path. |
| `12_full_cycle_craftext_training.ipynb` | Run CrafText rollout → replay evidence → token batch → masked PPO update as a compact full-cycle training example. |
| `17_sync_vllm_craftext_rollout.ipynb` | Run CrafText → MegaPrompts → Qwen chat template → sync vLLM generation → batched env step → replay. |
| `18_async_vllm_craftext_rollout.ipynb` | Run the same vLLM rollout through `GenerationBatch` async collection with bounded in-flight requests. |

Keep examples self-contained: fix seeds, avoid private data and do not import from `tests/`.
