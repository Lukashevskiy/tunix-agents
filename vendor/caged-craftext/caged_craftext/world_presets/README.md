# World Presets

World preset YAML uses only section-based schema:

```yaml
env:
  env_name: Craftax-Classic-Pixels-v1
  seed: 7

map:
  generator:
    name: box
    config:
      inner_size: 3
      perimeter_tree_prob: 1.0
      blocked_block: out_of_bounds
      floor_block: grass
      perimeter_block: tree

systems:
  static_env:
    map_size: 64
  spawn:
    disable_mob_spawns: true
  movement:
    behaviors:
      - solid_blocks
    rules:
      solid_out_of_bounds:
        value: true
        probability: 1.0
  reset:
    behaviors:
      - starting_inventory
      - starting_intrinsics
    starting_inventory:
      wood:
        value: 3
        probability: 1.0
  step:
    behaviors:
      - intrinsic_dynamics
      - instant_recovery
    rules:
      instant_sleep_recovery:
        value: true
        probability: 1.0
```

Rules:

- No top-level generator fields.
- No Python builtin presets; all presets are YAML files in this directory.
- Preset lookup is strict: `preset_name` maps directly to one YAML path in this directory.
- `extends` merges sections recursively before parsing.
- `env.env_name` validates compatibility with the active env, but does not override it.

Supported sections:

- `env`: `env_name`, `seed`
- `map.generator`: `name`, `config`
- `systems.static_env`: `map_size`
- `systems.spawn`: `disable_mob_spawns`
- `systems.movement`: `behaviors`, `rules`
- `systems.reset`: `behaviors`, `starting_inventory`, `starting_intrinsics`
- `systems.step`: `behaviors`, `intrinsic_rates`, `intrinsic_thresholds`, `rules`

Current generator configs:

- `box`: `inner_size`, `perimeter_tree_prob`, `blocked_block`, `floor_block`, `perimeter_block`
- `ring`: `inner_radius`, `outer_radius`, `blocked_block`, `floor_block`
