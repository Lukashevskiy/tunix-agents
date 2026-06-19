# CrafText Dataset Layout

Canonical dataset layout after refactor:

- `craftext/dataset/configs/<domain>/<difficulty>/<split>.yaml`
- `craftext/dataset/scenarious/<dataset_key>/instructions.py`
- Optional `test.py` is supported, but if it is missing loader falls back to `instructions.py`.

## Canonical dataset keys

- `building_line`
- `building_square`
- `building_star`
- `conditional_achievements`
- `conditional_placing`
- `explore`
- `localization_place`
- `achievements_wood`
- `all_building` (aggregated building tasks)
- `all_conditional` (aggregated conditional tasks)
- `all_localization` (aggregated localization tasks)
- `all` (aggregated full tasks)

## Loader behavior

`craftext.environment.scenarious.loader` is task-agnostic:

- Uses `dataset_key` directly as module path under `dataset/scenarious`
- Supports both styles without hardcoded task aliases:
  - `building_line`
  - `building.line`
- If `test: true` and `test.py` is absent, falls back to `instructions.py`
- Dataset package can be overridden via `CRAFTEXT_DATASET_PACKAGE` (useful for caged/forked package names)
