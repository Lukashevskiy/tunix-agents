"""Compatibility constants for CagedCrafText.

This layer mirrors the new CrafText constants while exposing ``IntEnum``
semantics expected by the older caged scenario datasets.
"""

from enum import IntEnum

from craftext.environment import craftext_constants as base_constants


def _as_int_enum(name: str, source_enum: object) -> IntEnum:
    return IntEnum(name, {member.name: member.value for member in source_enum})  # type: ignore[arg-type]


AchievementState = _as_int_enum("AchievementState", base_constants.AchievementState)
Scenarios = _as_int_enum("Scenarios", base_constants.Scenarios)
MediumInventoryItems = _as_int_enum("MediumInventoryItems", base_constants.MediumInventoryItems)
Achievement = _as_int_enum("Achievement", base_constants.Achievement)
InventoryItems = _as_int_enum("InventoryItems", base_constants.InventoryItems)
BlockType = _as_int_enum("BlockType", base_constants.BlockType)
TimeState = _as_int_enum("TimeState", base_constants.TimeState)
CrossType = _as_int_enum("CrossType", base_constants.CrossType)
MobType = _as_int_enum("MobType", base_constants.MobType)

base_path = base_constants.base_path
plans_path = base_constants.plans_path

__all__ = [
    "AchievementState",
    "Scenarios",
    "MediumInventoryItems",
    "Achievement",
    "InventoryItems",
    "BlockType",
    "TimeState",
    "CrossType",
    "MobType",
    "base_path",
    "plans_path",
]
