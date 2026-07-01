# Полновесный CLI слой

## Цель

Нам нужен один большой, но не распухший entrypoint: `tunix-craftext`. Он должен заменить
россыпь ad-hoc скриптов и стать стандартным способом запускать reproducibility checks,
environment smoke, prompt/model replay, Agentic GRPO training, benchmarks, docs/status sync и
audit. Главное требование: CLI управляет pipeline, но не прячет доменную логику внутри себя.

```text
profile/config -> cli command -> use-case service -> typed module contracts -> evidence artifact
```

CLI — это thin orchestration layer. Он отвечает за UX, валидацию входа, dry-run, вывод статуса,
exit codes и запись provenance. Environment adapters, MegaPrompts, Tunix/RLCluster, rollout,
learner, benchmarks и docs generators остаются в своих модулях.

## Принципы

1. **Profile-first.** Production команды принимают versioned YAML. Флаги могут переопределять
   только безопасные runtime knobs вроде `--output`, `--dry-run`, `--allow-cpu-smoke`.
2. **No implicit downloads.** Команды не скачивают модель/данные без явного subcommand или флага.
3. **Evidence by default.** Любой запуск пишет JSON/JSONL manifest до тяжёлой части: git,
   profile hash, vendor hash, package versions, device/backend, seed и output paths.
4. **Composable use-cases.** CLI вызывает функции уровня `use_cases/*`, а не импортирует
   private детали Tunix/CrafText напрямую.
5. **Typed config boundary.** Входные YAML превращаются в dataclass/Pydantic-like strict
   contracts до model allocation.
6. **Hardware-gated честность.** CPU/fake lanes, real CrafText lanes и accelerator lanes имеют
   разные команды и exit semantics; нельзя выдавать CPU smoke за performance baseline.
7. **Stable shell UX.** Команды должны быть короткими и предсказуемыми, с `--json` для CI и
   human-readable таблицами по умолчанию.

## Почему Typer, а не argparse

Сейчас scripts используют `argparse`. Для большого CLI лучше добавить optional runtime
dependency `typer`:

- nested commands читаются проще (`tunix-craftext train grpo ...`);
- автогенерация help богаче;
- легче делать shared options (`--profile`, `--output`, `--json`, `--dry-run`);
- тестирование через `CliRunner` компактнее.

Если не хотим новую dependency в base install, можно держать Typer в optional extra `cli`, но
для project CLI практичнее сделать его прямой lightweight dependency. Внутренние use-case
функции всё равно не зависят от Typer.

## Предлагаемая структура пакета

```text
src/tunix_craftext/
  cli/
    __init__.py
    app.py                 # root Typer app, entrypoint function
    common.py              # shared options, console/json output, exit codes
    profiles.py            # profile inspect/validate commands
    env.py                 # env smoke/reset/step commands
    prompts.py             # render prompt, decode action, inspect action catalog
    rollout.py             # collect fake/text/agentic rollouts
    train.py               # train grpo, future train ppo
    eval.py                # deterministic evaluation commands
    benchmark.py           # env/text/agentic performance commands
    docs.py                # dashboard/site/status/provenance sync
    audit.py               # repository/project audit command
  use_cases/
    __init__.py
    validate_profile.py
    collect_text_episode.py
    run_agentic_grpo.py
    benchmark_environments.py
    benchmark_text_pipeline.py
    sync_docs.py
```

`scripts/*.py` после миграции становятся тонкими backward-compatible wrappers:

```python
from tunix_craftext.cli.app import main

if __name__ == "__main__":
    main(["train", "grpo", *sys.argv[1:]])
```

Или удаляются после deprecation window, если уже есть стабильный `console_script`.

## Console entrypoint

В `pyproject.toml`:

```toml
[project.scripts]
tunix-craftext = "tunix_craftext.cli.app:main"
tcx = "tunix_craftext.cli.app:main"
```

`tcx` — короткий alias для частой разработки. В документации основное имя остаётся
`tunix-craftext`.

## Дерево команд

### 1. `profile`

Работа с versioned YAML без запуска тяжёлых операций.

```bash
tunix-craftext profile validate configs/training/grpo/qwen_agentic_local.yaml
tunix-craftext profile inspect configs/training/grpo/qwen_agentic_local.yaml --json
tunix-craftext profile evidence configs/training/grpo/qwen_agentic_local.yaml --output artifacts/runs/.../provenance.json
```

Ответственность:

- load strict schema (`MvpRunConfig`, `AgenticGrpoProfile`);
- показать resolved paths;
- посчитать profile/vendor SHA256;
- проверить наличие snapshot, но не загружать weights;
- вернуть non-zero exit code при schema drift.

### 2. `env`

Минимальные команды для среды и адаптеров.

```bash
tunix-craftext env smoke configs/env/smoke/tiny_craftext.yaml --steps 8
tunix-craftext env step configs/env/text/qwen_craftext.yaml --action LEFT --seed 7
tunix-craftext env inspect configs/env/text/qwen_craftext.yaml --json
```

Ответственность:

- доказать, что config создаёт CrafText/Caged adapter;
- вывести instruction, world preset, optional constraint, legal action catalog;
- записать small trajectory artifact при `--output`.

### 3. `prompt`

Граница MegaPrompts/text policy.

```bash
tunix-craftext prompt render configs/env/text/qwen_craftext.yaml --goal "collect wood"
tunix-craftext prompt decode configs/env/text/qwen_craftext.yaml --completion "<action>LEFT</action>"
tunix-craftext prompt replay artifacts/trajectories/qwen-craftext-latest.json
```

Ответственность:

- сформировать `PromptContext`;
- проверить, что user goal не теряется за scenario instruction;
- показать action mapping и strict decoder result.

### 4. `rollout`

Сбор траекторий без update.

```bash
tunix-craftext rollout random configs/env/text/qwen_craftext.yaml --horizon 32 --batch-size 8
tunix-craftext rollout text configs/env/text/qwen_craftext.yaml --snapshot artifacts/models/qwen25-05b-instruct
tunix-craftext rollout agentic configs/training/grpo/qwen_agentic_local.yaml --dry-run
```

Ответственность:

- собрать replay/trajectory JSONL;
- поддержать fake/deterministic backend для CPU CI;
- real Qwen/Tunix rollout только при явном snapshot/profile;
- для Agentic path — подготовить task stream и проверить group semantics.

### 5. `train`

Главный production path.

```bash
tunix-craftext train grpo configs/training/grpo/qwen_agentic_local.yaml
tunix-craftext train grpo configs/training/grpo/qwen_agentic_local.yaml --dry-run
tunix-craftext train grpo configs/training/grpo/qwen_agentic_local.yaml --allow-cpu-smoke
```

Ответственность:

- preflight profile/topology/Qwen tensor shape;
- записать provenance до model allocation;
- создать actor/rollout/reference через public Tunix `RLCluster`;
- запустить Agentic `GRPOLearner`;
- писать metrics JSONL, trajectories JSONL, checkpoints;
- non-zero exit при stale profile, missing snapshot, CPU-only accelerator-required run.

Будущий `train ppo` допустим только как отдельный experimental command:

```bash
tunix-craftext train ppo configs/ppo/...
```

Он не должен смешиваться с Agentic GRPO profile.

### 6. `eval`

Детерминированная оценка checkpoint/reference policy.

```bash
tunix-craftext eval checkpoint artifacts/runs/.../checkpoints/latest --tasks configs/eval/fixed_tasks.yaml
tunix-craftext eval reference configs/training/grpo/qwen_agentic_local.yaml --tasks configs/eval/fixed_tasks.yaml
```

Ответственность:

- fixed task list;
- success/reward/invalid-action/episode length;
- сравнение actor vs frozen reference;
- JSONL + summary table.

### 7. `benchmark`

Performance lanes с явным scope.

```bash
tunix-craftext benchmark env --matrix configs/env/benchmarks/*.yaml
tunix-craftext benchmark text configs/env/text/qwen_craftext.yaml --horizon 8 --repeats 20
tunix-craftext benchmark agentic configs/training/grpo/qwen_agentic_local.yaml --accelerator-required
```

Ответственность:

- разделять compile/warmup/steady-state;
- писать raw samples + median/p95;
- не сравнивать разные hardware/backend;
- помечать partial/failed child process evidence.

### 8. `docs`

Сайт, dashboard, task graph, provenance.

```bash
tunix-craftext docs sync
tunix-craftext docs build
tunix-craftext docs serve
tunix-craftext docs provenance --output artifacts/provenance.json
```

### 9. `verify`

Единые gates.

```bash
tunix-craftext verify unit
tunix-craftext verify golden
tunix-craftext verify full
```

`verify golden` должен быть CLI-эквивалентом `make verify-golden`.

### 10. `audit`

Аудит репозитория и проектных контрактов.

```bash
tunix-craftext audit repo
tunix-craftext audit architecture
tunix-craftext audit docs
```

Первый шаг может просто оборачивать существующий `.codex/skills/repository-audit/scripts/audit_repo.py`,
но итоговая форма должна жить в package/use-case layer, чтобы CI не зависел от локальной `.codex`.

## Use-case слой

Каждая команда вызывает один public use-case:

| CLI command | Use-case | Уже существующий код |
| --- | --- | --- |
| `profile validate/evidence` | `validate_agentic_grpo_profile()` | `grpo_profile.py` |
| `env smoke` | `run_environment_smoke()` | `runtime.py`, adapters |
| `prompt render/decode` | `render_prompt_preview()` | `prompts.py`, `text_policy.py` |
| `rollout text` | `collect_text_episode()` | `run_text_episode.py`, `episode.py` |
| `train grpo` | `run_agentic_grpo()` | `run_agentic_grpo.py`, `tunix/rlcluster_workload.py` |
| `benchmark env` | `run_environment_benchmark()` | `benchmark_environments.py` |
| `benchmark text` | `run_text_pipeline_benchmark()` | `benchmark_text_pipeline.py` |
| `docs build` | `build_docs_site()` | `generate_dashboard.py`, MkDocs |

Это даст нам тестируемость: CLI tests проверяют parsing/output/exit code, а use-case tests —
контракты pipeline.

## Output и exit codes

По умолчанию команды печатают короткий human summary:

```text
profile: qwen-agentic-craftext-local-smoke
status: valid
model snapshot: missing
provenance: artifacts/runs/.../provenance.json
```

`--json` печатает machine-readable объект. Ошибки:

| Code | Meaning |
| --- | --- |
| `0` | success |
| `2` | user/config/schema error |
| `3` | missing local artifact/model snapshot |
| `4` | hardware/backend requirement not satisfied |
| `5` | runtime/train failure |
| `6` | quality/audit gate failed |

## Расширяемость

Расширение идёт тремя способами:

1. **Новый profile schema**: отдельный loader + tests + migration doc.
2. **Новый backend/model**: model spec + preflight + factory; CLI command не меняет training
   semantics.
3. **Новый algorithm**: отдельный `train <algorithm>` command; не смешивать PPO critic path и
   Agentic GRPO actor/reference path.

Plugin-like registry можно добавить позже:

```python
TrainingCommand(name="grpo", profile_loader=..., runner=...)
BenchmarkCommand(name="env", runner=...)
```

Но первый MVP лучше сделать явно: меньше магии, легче отлаживать.

## TDD plan для CLI

1. `test_cli_profile_validate_accepts_grpo_profile`.
2. `test_cli_profile_validate_rejects_unknown_keys`.
3. `test_cli_profile_evidence_writes_manifest_before_snapshot_check`.
4. `test_cli_train_grpo_dry_run_does_not_import_tunix_heavy_modules`.
5. `test_cli_verify_golden_invokes_expected_checks_with_fake_runner`.
6. `test_cli_json_output_is_stable`.
7. `test_scripts_are_backwards_compatible_wrappers` на время миграции.

CLI tests не должны загружать Qwen weights и не должны требовать accelerator.

## Миграция

### Slice 1: scaffold

- добавить `typer`;
- создать `tunix_craftext.cli.app`;
- добавить `profile validate`, `profile evidence`, `verify golden`;
- добавить `project.scripts`;
- покрыть unit tests.

### Slice 2: scripts as use-cases

- вынести `run_agentic_grpo.py` heavy logic в `use_cases/run_agentic_grpo.py`;
- CLI `train grpo` вызывает use-case;
- старый script становится wrapper.

### Slice 3: env/prompt/rollout

- добавить `env smoke`, `prompt render`, `rollout random/text`;
- унифицировать output artifact schema.

### Slice 4: benchmark/docs/audit

- добавить `benchmark env/text`, `docs build/sync/serve`, `audit repo`;
- `make` начинает вызывать CLI, а не наоборот.

### Slice 5: accelerator gates

- `train grpo --dry-run`;
- `rollout agentic`;
- `eval checkpoint/reference`;
- hardware-gated integration tests.

## Риски и предохранители

- **Риск:** CLI станет вторым местом бизнес-логики.  
  **Предохранитель:** вся логика в use-case modules; CLI tests проверяют только UX.

- **Риск:** один command начнёт неявно скачивать модели.  
  **Предохранитель:** downloads только в отдельном future command, golden path требует existing snapshot.

- **Риск:** argparse scripts и новый CLI разойдутся.  
  **Предохранитель:** scripts превращаются в wrappers или помечаются deprecated.

- **Риск:** Typer dependency осложнит минимальную установку.  
  **Предохранитель:** если решим не добавлять в base, сделать `cli` extra и оставить Makefile commands
  через `uv run python -m tunix_craftext.cli.app`.

## Первый acceptance gate

Слой считается спроектированным и готовым к реализации, когда:

- эта страница опубликована в docs site;
- `docs/plan.md` содержит CLI implementation slice;
- первый implementation PR ограничен `profile` + `verify` командами;
- ни одна новая CLI команда не импортирует Qwen/Tunix heavy modules на `--help`, `profile validate`
  или `verify golden`.
