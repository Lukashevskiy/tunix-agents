# ADR 0002: первая модель для Tunix bridge

## Статус

Принято для smoke-профиля; PPO workload profile ожидает явного выбора весов и лицензии.

## Решение

Первым implementation target является **Gemma 3 270M instruction** через native Tunix
model API. Это наименьшая документированная text instruction architecture в pinned Tunix
revision и позволяет проверить tokenizer, sampling, action parser и resource placement без
подмены model layer фиктивным объектом.

Первым PPO workload profile является **Qwen2.5-0.5B-Instruct**: он фигурирует в PPO
материалах Tunix и ближе к реальному LLM-agent policy. Его загрузка не начнётся автоматически:
она требует выбранного источника весов, лицензии, storage path и accelerator budget.

## Последствия

- Model selection входит в versioned config; не будет hard-coded в adapter.
- `TunixPolicyAdapter` сначала доказывает prompt/tokenizer/sample/logprob/value parity на
  Gemma smoke profile.
- Qwen profile добавляется отдельной compatibility fixture, не заменяет smoke profile.
- ONNX не является training runtime: JAX/Flax/Tunix остаётся source of truth для обучения;
  ONNX возможен позднее только как inference export target.
