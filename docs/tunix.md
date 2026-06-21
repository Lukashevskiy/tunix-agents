# Tunix integration boundary

Tunix is deliberately an optional dependency. The project pins the official
[`google/tunix`](https://github.com/google/tunix) revision in
`compatibility/tunix.yaml`; it must not use the
unrelated `tunix==0.0.0` PyPI distribution.

Install the bridge environment explicitly:

```bash
pyenv exec python -m uv sync --all-extras
```

The public boundary is intentionally narrow: `PPOConfig` and `PPOLearner`.
Our environment/prompt/replay contracts remain framework-neutral. The next
implementation task is `TunixPolicyAdapter`, which maps rendered prompts and
tokenized samples to Tunix sampling/logprob/value APIs and proves parity against
a direct Tunix invocation before it is allowed into a real rollout.

The base environment does not import Tunix. This keeps `make test`, CrafText
collection and Flax/Optax smoke learning independent of the heavyweight model
and tokenizer stack while preserving an exact, reproducible bridge source.
