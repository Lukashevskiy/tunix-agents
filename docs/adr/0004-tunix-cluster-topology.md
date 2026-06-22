# ADR 0004: Tunix владеет исполнением на mesh, проект владеет топологией ролей

## Статус

Принято.

## Контекст

Tunix использует JAX `Mesh` и умеет загружать либо ресшардировать модель на
целевой mesh. Его `RLCluster` принимает явное отображение `role_to_mesh` для
`actor`, `rollout`, `critic` и `reference`; этим задаются colocated и
disaggregated варианты обучения.

В раннем Qwen smoke был добавлен `QwenTunixBackend` с single-device sampler.
Он доказывает, что локальные Qwen-веса, tokenizer и публичный Tunix sampler
реально исполняются. Он **не** является multi-GPU архитектурой и не должен
подменять `RLCluster` в workload-профиле.

Проверка named `fsdp/tp` sampler-пути на pinned Tunix показала несовместимость
при Qwen generation. Мы не меняем vendor-код и не прячем её локальным патчем.

## Решение

- Производственный Tunix training/rollout строится через `RLCluster` и
  versioned `ClusterConfig.role_to_mesh`; проект задаёт доступные устройства и
  роль каждой части workload, Tunix выполняет загрузку, ресшардирование и JAX
  execution.
- `ResourceConfig` становится единственным местом, где выбираются topology,
  named axes и placement policy. Никакой adapter не выбирает GPU самостоятельно.
- `QwenTunixBackend` остаётся только локальным smoke/inference backend с
  provenance `tunix-single-device`. Его результаты нельзя использовать как
  показатель multi-device scale-up.
- Следующая реализация для Qwen: typed `ModelAdapter` с корректным chat-template
  и token/logprob bridge; после этого — `RLCluster` integration fixture на
  доступном accelerator mesh.
- Любая работа вокруг named Qwen sampler требует отдельного воспроизводимого
  upstream issue и compatibility test. До принятого upstream-исправления не
  форкаем Tunix и не вносим vendor patch.

## Последствия

- Tunix остаётся владельцем model sharding, offload и распределённого execution;
  у нас остаются явные, тестируемые contracts конфигурации и среды.
- На одном GPU/macOS разработка не имитирует масштабирование: hardware-gated
  multi-device integration тест будет отдельной CI/performance lane.
- structured action decoding — независимый слой. Неудачный свободный completion
  Qwen должен учитываться как decode failure/fallback metric, а не маскироваться
  изменением mesh.
