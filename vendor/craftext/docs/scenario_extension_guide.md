# CrafText: Гайд по Расширению Сценариев

Этот документ описывает, как расширять текущую архитектуру сценариев:
- добавлять новые датасеты (например, `constrained`);
- прокидывать новые конфиги и пути;
- настраивать `manager` и `loader`;
- собирать wrapper и инициализировать среду под свою задачу.

## 1. Коротко про текущий пайплайн

Путь данных:
1. `config_name` -> `CraftextScenariosConfigLoader.load_config(...)`  
   Файл: `craftext/environment/scenarious/loader.py`
2. `load_scenarios(config)` импортирует модуль `instructions.py` по `dataset_key` и берет словарь по `subset_key`.
3. `BaseScenarioDataHandler` собирает плоские строки (`ScenarioRows`) и отдает в фабрику данных.
4. `scenario_data_pipeline.py` выполняет последовательность компонентов:
   - `build_base_payload`
   - опционально `compute_embeddings`
   - `finalize_raw_payload` или `finalize_encoded_payload`
5. `JaxScenarioDataHandler` переводит данные в JAX через `jax_representation_class`.
6. Wrapper (`RawInstructionWrapper` или `EncodedInstructionWrapper`) использует `scenario_data_jax` во время `reset/step`.

## 2. Формат датасета `instructions.py`

Минимально нужно:
- переменная с именем `subset_key` (например, `easy`);
- значение: `dict[str, dict]`;
- внутри каждой записи используются поля:
  - `instruction: str`
  - `instruction_paraphrases: list[str]` (можно пустой)
  - `scenario_checker`
  - `arguments`

Пример:

```python
# craftext/dataset/scenarious/constrained/instructions.py
from craftext.environment.craftext_constants import Scenarios
from craftext.environment.scenarious.checkers.target_state import TargetState

def target_any() -> TargetState:
    return TargetState()  # пример-заглушка, подставьте свой target state

easy = {
    "INSTR_1": {
        "instruction": "Build a 3-block line without using stone.",
        "instruction_paraphrases": [
            "Make a line of length 3 and do not use stone."
        ],
        "scenario_checker": Scenarios.BUILD_LINE,
        "arguments": target_any(),
    }
}
```

## 3. Добавление нового конфига

Файл:
`craftext/dataset/configs/constrained/easy/train.yaml`

Пример:

```yaml
dataset_key: constrained
subset_key: easy
base_environment: Craftax-Classic-Pixels-v1-Text
use_parafrases: true
test: false
use_constraints_parafrases: false  # optional
world_preset: classic              # optional
```

Запуск с таким конфигом делается через:
- `config_name="constrained/easy/train"` или
- эквивалентные варианты, поддерживаемые loader (`_`, `.`, прямой путь).

## 4. Как loader резолвит модули и пути

В `loader.py` уже есть:
- поиск конфигов через `_find_config_path(...)`;
- резолв `dataset_key` в модуль:
  - `building_line` -> пробует `building.line` и `building_line`;
  - поддержка `test`-режима (`...<dataset>.test`).
- нормализация `base_environment`:
  - `Classic`, `Craftax-Classic-*` -> classic world family;
  - `Craftax-*` без `Classic` -> full world family;
  - optional alias `world_presets` также принимается и маппится в `world_preset`.

Чтобы использовать отдельный пакет датасетов (не `craftext.dataset`):
1. Сделайте наследника `CraftextScenariosConfigLoader`.
2. Переопределите `get_config_path`/`load_config` так, чтобы передавать свой `dataset_package`.
3. В менеджере переопределите `_load_config`.

Скелет:

```python
from craftext.environment.scenarious.loader import (
    CraftextScenariosConfigLoader,
    _find_config_path,
    _load_config_from_path,
    ScenariosConfig,
)

class MyScenariosConfigLoader(CraftextScenariosConfigLoader):
    @staticmethod
    def get_config_path(config_name: str):
        return _find_config_path("my_project.dataset", config_name)

    @staticmethod
    def load_config(config_name: str) -> ScenariosConfig:
        p = MyScenariosConfigLoader.get_config_path(config_name)
        return _load_config_from_path(p, "my_project", config_name)
```

## 5. Как подстроить manager под constrained-текст

Обычно делают отдельный хендлер-наследник и переопределяют `_collect_scenario_rows`.

Пример: использовать отдельное поле `constrained_instruction` при включенном флаге.

```python
from craftext.environment.scenarious.manager import BaseScenarioDataHandler
from craftext.environment.scenarious.scenario_data_pipeline import ScenarioRows

class ConstrainedScenarioDataHandler(BaseScenarioDataHandler):
    def _collect_scenario_rows(self) -> ScenarioRows:
        rows = super()._collect_scenario_rows()

        # Пример: заменить базовые инструкции на constrained-вариант,
        # если в исходных entry он есть.
        new_instructions = []
        for name in rows.scenario_names:
            base_name = name.replace("_PARA", "")
            entry = self.all_scenario[base_name]
            new_instructions.append(entry.get("constrained_instruction", entry["instruction"]))

        rows.instructions_list = new_instructions
        return rows
```

Если нужна более чистая схема: добавьте отдельный pipeline-компонент в
`craftext/environment/scenarious/scenario_data_pipeline.py` и зарегистрируйте его в последовательности фабрики.

## 6. Параметризация финального encoded payload

В `scenario_data_pipeline.py` компонент `FinalizeEncodedPayloadComponent` уже принимает `encoded_payload_cls`.

Это позволяет использовать не только `EncodedScenarioData`, но и ваш собственный класс, если он принимает аргументы:
- `instructions_list`
- `scenario_checker`
- `arguments`
- `scenario_names`
- `embeddings_list`

Подключение:

```python
from craftext.environment.scenarious.scenario_data_pipeline import create_encoded_scenario_data_factory

factory = create_encoded_scenario_data_factory(
    processor_provider=lambda: self.scenario_processor,
    encoded_payload_cls=MyEncodedScenarioData,
)
```

## 7. Сборка wrapper под свои нужды

### 7.1 Raw-поток

```python
import jax
from craftax.craftax_env import make_craftax_env_from_name
from craftext.environment.scenarious.manager import JaxScenarioDataHandler, DefaultJAXRepresentation
from craftext.environment.scenarious.processors import RawProcessor
from craftext.environment.scenarious.instruction_transformers import DefaultInstructionTransformer
from craftext.environment.craftext_wrapper import RawInstructionWrapper

env = make_craftax_env_from_name("Craftax-Classic-Pixels-v1", auto_reset=False)
handler = JaxScenarioDataHandler(
    scenario_processor=RawProcessor,
    instruction_transformer=DefaultInstructionTransformer,
    config_name="constrained/easy/train",
    jax_representation_class=DefaultJAXRepresentation,
)
wrapper = RawInstructionWrapper(env, handler)

rng = jax.random.PRNGKey(0)
obs, state = wrapper.reset(rng, env.default_params)
```

### 7.2 Encoded-поток

Важно: `BaseScenarioDataHandler` создает процессор как `scenario_processor()`, то есть класс должен иметь `__init__` без аргументов.  
Для модели энкодера обычно делают адаптер-класс.

```python
from craftext.environment.scenarious.processors import EncodedProcessor

class MyEncodedProcessor(EncodedProcessor):
    def __init__(self):
        super().__init__(encode_model=build_or_load_model())
```

Дальше:
- `JaxScenarioDataHandler(... scenario_processor=MyEncodedProcessor, ...)`
- `jax_representation_class=EncodedJAXRepresentation`
- wrapper: `EncodedInstructionWrapper`.

## 8. Частые ошибки

1. `subset_key` не совпадает с именем переменной в `instructions.py`.
2. `dataset_key` не резолвится в модуль (ошибка импорта).
3. `EncodedProcessor` передан как класс с обязательными аргументами конструктора.
4. Структура сценария не содержит `scenario_checker`/`arguments`.
5. Для encoded-потока выбран `RawInstructionWrapper` (или наоборот).

## 9. Мини-чеклист перед запуском

1. Есть файл конфига в `craftext/dataset/configs/...`.
2. `dataset_key`/`subset_key` валидны.
3. `instructions.py` экспортирует нужный словарь.
4. Выбран правильный processor + JAX representation + wrapper.
5. `config_name` соответствует реальному пути и формату loader.
