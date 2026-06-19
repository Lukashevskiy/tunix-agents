# CrafText: Архитектура Проекта

Этот документ описывает текущую архитектуру `CrafText`:
- как устроены директории и модули;
- за что отвечает каждый слой;
- как модули связаны между собой;
- как данные проходят путь от конфига до шага в среде.

Документ ориентирован на текущий код в `craftext/`.

## 1. Верхнеуровневая структура

```text
CrafText/
  craftext/
    dataset/
      configs/                  # YAML-конфиги наборов сценариев
      scenarious/               # сами сценарии (instructions.py / test.py)
    environment/
      craftext_constants.py     # enum'ы и общие константы домена
      craftext_wrapper.py       # wrapper'ы над env + reward/check logic
      craftext_play.py          # локальный интерактивный раннер
      encoders/                 # текстовые энкодеры (DistilBERT и т.д.)
      scenarious/
        base.py                 # абстракции для scenario handler
        loader.py               # резолв и загрузка конфигов/датасетов
        manager.py              # orchestration сценариев + JAX conversion
        processors.py           # raw/encoded процессоры инструкций
        instruction_transformers.py
        scenario_data_pipeline.py
        encoded_support.py
        checkers/               # проверка выполнения задач
      states/
        state.py                # extraction для non-classic режима
        state_classic.py        # extraction для classic режима
  docs/
    scenario_extension_guide.md
    project_architecture.md
```

## 2. Слои и их ответственность

### 2.1 Dataset Layer (`craftext/dataset`)

Назначение:
- хранить декларативные данные сценариев и их конфигурации.

Содержимое:
- `configs/**/*.yaml`: какой датасет загрузить (`dataset_key`), какой subset (`subset_key`), режимы (`use_parafrases`, `test`, и т.д.).
- `scenarious/**/instructions.py`: словари сценариев, где каждый сценарий содержит `instruction`, `scenario_checker`, `arguments`, `instruction_paraphrases`.

Важно:
- dataset-слой не выполняет бизнес-логику проверки; он только поставляет данные.

### 2.2 Scenario Runtime Layer (`craftext/environment/scenarious`)

Назначение:
- загрузить конфиг;
- импортировать нужный модуль с инструкциями;
- материализовать сценарии в runtime-структуры;
- при необходимости посчитать эмбеддинги;
- подготовить JAX-структуры для fast-step.

Ключевые модули:
- `loader.py`: валидирует YAML и резолвит путь/модуль сценариев.
- `manager.py`: orchestration процесса, собирает `scenario_data` и `scenario_data_jax`.
- `scenario_data_pipeline.py`: фабрика-пайплайн для сборки payload'ов.
- `processors.py`: raw/encoded preprocessing инструкций.
- `instruction_transformers.py`: трансформация текста инструкций (например, plans).
- `encoded_support.py`: структуры и конвертер для encoded-потока.
- `checkers/*`: проверка выполнения цели по `TargetState`.

### 2.3 Environment Integration Layer (`craftext/environment`)

Назначение:
- связать Craftax env и scenario runtime;
- вести состояние текстовой задачи (`idx`, `checker_id`, `target_state`);
- считать done/reward через checker-функции.

Ключевые модули:
- `craftext_wrapper.py`: `BaseInstructionWrapper`, `RawInstructionWrapper`, `EncodedInstructionWrapper`.
- `states/state.py`, `states/state_classic.py`: адаптеры env state в формат для checkers.
- `craftext_constants.py`: enum'ы сценариев, блоков, достижений.
- `craftext_play.py`: пример инициализации среды и ручной loop.

## 3. Связность модулей

## 3.1 Основной граф зависимостей

```text
dataset/configs/*.yaml
        |
        v
loader.py (ScenariosConfig + load_scenarios)
        |
        v
manager.py (BaseScenarioDataHandler / JaxScenarioDataHandler)
        |
        +--> scenario_data_pipeline.py (factory + components)
        |
        +--> processors.py (RawProcessor / EncodedProcessor)
        |
        +--> instruction_transformers.py
        |
        +--> encoded_support.py (для encoded payload/JAX)
        |
        v
scenario_data + scenario_data_jax
        |
        v
craftext_wrapper.py
        |
        +--> states/state*.py (extract GameData)
        +--> checkers/registry.py (lax.switch -> checker fn)
        v
env.step/reset loop
```

### 3.2 Кто кого импортирует (практически)

- `craftext_wrapper.py` импортирует:
  - `JaxScenarioDataHandler` из `manager.py`;
  - `GameData/GameDataClassic` из `states/*`;
  - `CHECKER_FUNCTIONS` из `checkers/registry.py`.
- `manager.py` импортирует:
  - `ScenariosConfigLoader/load_scenarios` из `loader.py`;
  - pipeline API из `scenario_data_pipeline.py`;
  - processors/transformers/TargetState.
- `loader.py` импортирует dataset-модули динамически по `dataset_key`.

## 4. Потоки выполнения

### 4.1 Raw-поток (без эмбеддингов)

1. Создается `JaxScenarioDataHandler` с `RawProcessor` и `DefaultJAXRepresentation`.
2. `loader.load_config(config_name)` читает YAML.
3. `load_scenarios(config)` импортирует `instructions.py` и берет `subset_key`.
4. `manager._collect_scenario_rows()` разворачивает base + paraphrases.
5. `scenario_data_pipeline` исполняет компоненты:
   - `build_base_payload`
   - `finalize_raw_payload`
6. `DefaultJAXRepresentation` создает:
   - `scenario_checker` как JAX array;
   - `arguments` как stacked `TargetState`.
7. `RawInstructionWrapper.reset/step` использует эти данные в runtime.

### 4.2 Encoded-поток (с эмбеддингами)

1. `JaxScenarioDataHandler` создается с `EncodedProcessor`-совместимым классом.
2. Pipeline выполняет:
   - `build_base_payload`
   - `compute_embeddings` (через `processor.process`)
   - `finalize_encoded_payload`
3. `EncodedJAXRepresentation` преобразует payload в JAX, включая `embeddings_list`.
4. `EncodedInstructionWrapper` читает `scenario_data_jax.embeddings_list[idx]`.

## 5. Детализация ключевых модулей

### 5.1 `loader.py`

Ответственность:
- схема и валидация конфига (`CONFIG_SCHEMA`);
- резолв имен конфига в путь (`_find_config_path`);
- динамический импорт датасета (`_import_scenario_module`);
- получение сценарного словаря по `subset_key`.

Инварианты:
- все ключи YAML должны быть в схеме;
- обязательные поля не пустые;
- `subset_key` обязан существовать как имя переменной в `instructions.py`.

### 5.2 `manager.py`

Ответственность:
- orchestration жизненного цикла сценариев:
  - config -> raw scenarios -> rows -> payload -> jax payload.

Ключевые классы:
- `BaseScenarioDataHandler`: загрузка и материализация сценарных данных.
- `JaxScenarioDataHandler`: дополнительный этап `scenarios_to_jax`.
- `DefaultJAXRepresentation`: default converter для raw payload.

### 5.3 `scenario_data_pipeline.py`

Ответственность:
- изолировать сборку payload'ов в конфигурируемый pipeline:
  - компоненты;
  - реестр компонентов;
  - `ComponentSpec`;
  - generic-фабрики `create_raw_scenario_data_factory` / `create_encoded_scenario_data_factory`.

Плюсы:
- расширение без разрастания `manager.py`;
- параметризация финального класса payload для encoded ветки;
- более строгая типизация фабрик по возвращаемому payload.

### 5.4 `checkers/`

Ответственность:
- pure-функции проверки достижения цели из `(game_data, target_state)`.

Центральная точка:
- `checkers/registry.py`: сопоставление `Scenarios.*` -> checker callable.

### 5.5 `craftext_wrapper.py`

Ответственность:
- интеграция сценариев в env loop:
  - выбор инструкции при reset;
  - вычисление `instruction_done` при step;
  - reward scaling/termination;
  - формирование `TextEnvState`.

Критичные зависимости:
- `CHECKER_FUNCTIONS` (порядок должен соответствовать enum `Scenarios.value`).
- `scenario_data_jax` должен иметь корректный тип под wrapper (raw/encoded).

## 6. Типы данных и контракты

### 6.1 Сценарный payload

Базовый структурный контракт (`ScenarioDataPayload`):
- `instructions_list`
- `scenario_checker`
- `arguments`
- `scenario_names`

Raw реализация:
- `BaseScenarioData`.

Encoded реализация:
- `EncodedScenarioData` (+ `embeddings_list`).

### 6.2 JAX payload

Raw JAX:
- `BaseScenarioDataJAX` (`scenario_checker`, `arguments`).

Encoded JAX:
- `EncodedScenarioDataJAX` (`scenario_checker`, `arguments`, `embeddings_list`).

## 7. Изменяемость и extension points

### 7.1 Добавление нового типа задач

Нужно затронуть:
1. `craftext_constants.py`: новый `Scenarios` enum.
2. `checkers/`: новый checker + регистрация в `registry.py`.
3. `TargetState` в `checkers/target_state.py`: новый под-state.
4. dataset `instructions.py`: `scenario_checker` и `arguments` нового типа.

### 7.2 Добавление нового способа финализации payload

Варианты:
- добавить новый `ScenarioBuildComponent`;
- зарегистрировать через `PipelineScenarioDataFactory.register_component(...)`;
- включить в `component_sequence` нужного handler/factory.

### 7.3 Подмена encoded payload класса

Используется `create_encoded_scenario_data_factory(..., encoded_payload_cls=...)`.
Класс должен принимать поля:
- `instructions_list`, `scenario_checker`, `arguments`, `scenario_names`, `embeddings_list`.

## 8. Связность и границы ответственности

Слабая связность (желательная):
- dataset не знает про env runtime;
- checkers не знают про loader/manager;
- pipeline не зависит от конкретных dataset keys.

Более сильная связность (осознанная):
- wrapper зависит от структуры `scenario_data_jax`;
- `Scenarios` enum завязан на `registry.py` и данных в датасете;
- `TargetState` должен быть согласован с checker-функциями.

## 9. Типичный сценарий интеграции end-to-end

1. Создать YAML в `dataset/configs/...`.
2. Создать/обновить `instructions.py` с нужным `subset_key`.
3. Поднять `JaxScenarioDataHandler` (raw или encoded).
4. Собрать `RawInstructionWrapper` или `EncodedInstructionWrapper`.
5. Запустить `reset/step` цикл.

См. также:
- `docs/scenario_extension_guide.md` (практический гайд расширений).

## 10. Технические заметки

1. В коде встречается историческое написание `scenarious`; при рефакторинге лучше менять атомарно.
2. При локальном запуске возможны ошибки backend JAX CUDA (`No visible GPU devices`); для CPU-проверок используйте `JAX_PLATFORMS=cpu`.
3. `craftext/environment/representation.py` удален как неиспользуемый слой; источники truth для representation теперь:
   - `manager.py` (raw JAX representation),
   - `encoded_support.py` (encoded JAX representation).
