# Правило поставки каждого изменения

Ни один фикс, feature или изменение публичного поведения не считается завершённым, пока не
оставил проверяемый след во всех применимых слоях.

## Definition of Done

1. **Аудит:** выполнить read-only audit и разобрать findings до commit.
2. **Тесты:** добавить или обновить тест, который сначала бы падал на старом поведении; запустить
   unit и применимый integration suite.
3. **Профилирование:** для JAX hot path, алгоритма, collection, model conversion или изменения
   batch/mesh запустить performance test. Сохранить JSON в `artifacts/benchmarks/` с commit,
   config hash, seed, device, warmup и результатом. Если perf неприменим, явно написать почему в
   commit/PR и не заявлять performance improvement.
4. **Документация:** обновить архитектуру, ADR, usage или quality docs, если пользовательский
   контракт или проектное решение изменилось.
5. **Статус:** обновить checkbox в `docs/plan.md` и capability в `docs/project_status.json`.
   `ready` означает: есть реализация и проходящий тест.
6. **Commit и сайт:** сделать осмысленный commit и запустить `make docs`; dashboard прочитает
   новый Git revision, roadmap и benchmark artifacts.

## Стандартные команды

```bash
make audit
make verify
make perf  # обязательно для изменений горячего пути
```

`make verify` запускает audit, unit/integration tests и строгую сборку сайта. Он не подменяет
performance evidence: профилировать на известном hardware следует отдельно.
