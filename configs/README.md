# Configuration layout

`configs/` is organized by responsibility, not by project history.

Canonical paths:

- `env/` — CrafText/Caged-CrafText environment, prompt and policy profiles.
  - `env/smoke/` — small deterministic configs for unit tests and notebooks.
  - `env/text/` — normal text-agent rollout configs.
  - `env/manual/` — human-in-the-loop/manual-control scenarios.
  - `env/benchmarks/` — environment performance matrix configs.
- `inference/` — generation backend contracts.
  - `inference/vllm/` — sync/async vLLM rollout profiles.
- `models/` — model cards/profiles and licence/resource metadata.
- `tunix/topology/` — Tunix role meshes and resource placement.
- `training/` — end-to-end training profiles.
  - `training/grpo/` — Agentic GRPO profiles binding env, inference, topology, model and evidence paths.

Compatibility paths:

Older paths such as `configs/mvp/...`, `configs/generation/...`,
`configs/topology/...` and `configs/grpo/...` are symlinks kept only so old
commands and notebooks fail less abruptly. New code, docs and profiles should
use canonical paths from `configs/index.yaml`.

Rule of thumb:

If a file configures one module, put it under that module's folder. If it wires
several modules into a runnable experiment, put it under `training/<algorithm>/`.
