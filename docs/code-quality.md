# Типы и docstrings

Типы, shapes и docstrings — часть контракта модуля. В JAX это защищает от ошибок, которые
иначе проявляются лишь после compilation или на accelerator.

## Обязательные правила

- Каждая публичная функция, метод и класс имеет type annotations для параметров и результата.
- Каждый публичный API имеет Sphinx-compatible docstring: назначение, shape/оси/dtype для
  массивов, единицы или semantics значений, PRNG ownership, mutation policy, raises и return.
  Использовать field-list: `:param name:`, `:returns:`, `:raises Error:`; типы берутся из Python
  annotations через `sphinx-autodoc-typehints`, не дублируются в prose.
- Не использовать `Any` в domain contracts, PyTree boundaries, config или public return types.
  Вместо него выбирать `TypeVar`, `Protocol`, dataclass/NamedTuple, `Mapping`, конкретный array
  alias или небольшой typed adapter.
- `Any` допустим только на недоверенной внешней границе (например, raw third-party payload) и
  должен быть немедленно преобразован/проверен внутри одного adapter. Такой случай документируется.
- Не подменять отсутствие знания ложным узким типом: если структура расширяема, выразить это
  через bounded protocol или generic parameter.

## JAX contracts

В docstring явно указывать ведущие оси (например `[T, B, ...]`), static vs dynamic fields,
sharding axis, dtype, допустимые shapes и numerical tolerance. Pure/JIT functions не должны
полагаться на скрытый host state или неявный global RNG.

JAX contracts сами регистрируют свои immutable dataclasses как PyTrees в module of definition.
Не выполнять registration в consumer module и не использовать NumPy host conversion внутри
JIT-visible methods: shape validation — отдельная host-side boundary операция.

Для численных полей использовать прямые JAX-аннотации: `jax.typing.ArrayLike` на внешнем
принимающем boundary и `jax.Array` в нормализованных публичных контрактах. Не вводить общий
`ArrayT` ради совместимости с массивами, которые не переживут `jax.jit`.

## Review checklist

Перед commit проверить: добавлены ли annotations и docstrings; может ли вызывающий понять shape
без чтения implementation; не проник ли `Any` глубже boundary adapter; обновлён ли тест для
неверных shapes/dtypes; отражён ли изменённый контракт в architecture/docs.

## API reference

`make api-docs` запускает Sphinx в warnings-as-errors режиме и создаёт API reference в
`site/api/`. Конфигурация лежит в `sphinx/` и документирует public signatures из `src/`.
