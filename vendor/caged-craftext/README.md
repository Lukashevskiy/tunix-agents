# CagedCrafText

`CagedCrafText` is the canonical root package for constraint-aware CrafText scenarios.

## Layout

- `environment/`: caged wrappers, checkers, scenario loaders, and world preset integration.
- `dataset/configs/`: YAML config entrypoints for caged scenario families.
- `dataset/scenarious/`: scenario definitions used by the caged loaders.
- `world_presets/`: local preset YAML files used by `CagedCrafText.environment.world_presets`.

## API rules

- Import this package as `CagedCrafText`, not `caged_craftext`.
- Use `CagedCrafText.environment.scenarious.checkers.constrained_target_state` directly for caged target-state types.
- `CrafText` remains the base environment package and dependency source.

## Manual Play

- Use `caged_craftext_play` for interactive constraint-aware play.
- Default runtime profile:
  - config: `budget/achievements/easy/explore_energy_above_8`
  - world preset: `caged_craftext_play`
- The default preset gives a small boxed map with trees, instant sleep recovery, and `1` energy drain per action.

## Verification

All `instructions.py` modules under `dataset/scenarious` are expected to import successfully with:

```bash
MPLCONFIGDIR=/tmp/mpl JAX_PLATFORMS=cpu PYTHONPATH=/path/to/repo:/path/to/repo/CrafText python -m ...
```
