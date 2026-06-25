# Карточка эксперимента

Каждый запуск создаёт `artifacts/runs/<run-id>/`:

```text
config.resolved.yaml    # полный config, включая vendor revision
metadata.json           # Git, hardware, packages, seed, timestamps
metrics.jsonl           # train/eval/throughput метрики
validation_trajectories.jsonl # ссылки на полные val trajectories
trajectory/             # компактный replay + rendered prompt reference
checkpoints/            # Orbax checkpoints + schema version
benchmark.json          # если это perf run
```

`run-id` должен включать short Git revision и config hash. Документационный сайт читает
только опубликованные, явно выбранные artifacts; он не сканирует личные или тяжёлые
checkpoints. Так статический сайт остаётся быстрым и не раскрывает случайные данные.

Перед сборкой релизного сайта CI запускает `python scripts/export_provenance.py` и
публикует получившийся JSON рядом с artifacts. Так commit, dirty flag, Python и platform
берутся из Git при сборке, а не из хрупкого шаблонного плагина документации.
