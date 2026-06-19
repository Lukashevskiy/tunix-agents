# ADR-0001: vendored environments, native training core

**Status:** accepted

## Context

CrafText и CagedCrafText уже содержат JAX-ориентированную environment логику, а MegaPrompts
— ценные assets. VERL/Ray orchestration не является необходимой частью их семантики.

## Decision

Копируем три исходника с их LICENSE в `vendor/` как неизменяемую baseline. Новый код живёт
только в `src/tunix_craftext`, общается с vendor через adapters и не импортирует VERL. Любое
изменение vendor оформляется как осознанное upstream sync с manifest и parity tests.

## Consequences

Переход можно профилировать и откатывать независимо. Цена — необходимость поддерживать
adapter и явно фиксировать версии, что лучше, чем неявный дрейф поведения среды.
