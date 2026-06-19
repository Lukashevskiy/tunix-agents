from __future__ import annotations

from collections.abc import Iterable
from typing import Any


ACTION_DESCRIPTIONS = {
    "NOOP": "do nothing",
    "LEFT": "move west on flat ground",
    "RIGHT": "move east on flat ground",
    "UP": "move north on flat ground",
    "DOWN": "move south on flat ground",
    "DO": "multiuse action to collect material, drink from lake, mine, chop trees, and hit the creature in front",
    "SLEEP": "sleep in a safe location when energy is low",
    "PLACE_STONE": "place a stone in front when nothing is in front; requires 1 stone",
    "PLACE_TABLE": "place a crafting table in front when nothing is in front; requires 2 wood",
    "PLACE_FURNACE": "place a furnace in front when nothing is in front; requires 4 stone",
    "PLACE_PLANT": "place a plant in front; requires 1 sapling",
    "MAKE_WOOD_PICKAXE": "craft a wood pickaxe; requires standing next to a table and 1 wood",
    "MAKE_STONE_PICKAXE": "craft a stone pickaxe; requires standing next to a table, 1 wood, and 1 stone",
    "MAKE_IRON_PICKAXE": "craft an iron pickaxe; requires standing next to a table and furnace, plus 1 wood, 1 coal, and 1 iron",
    "MAKE_WOOD_SWORD": "craft a wood sword; requires standing next to a table and 1 wood",
    "MAKE_STONE_SWORD": "craft a stone sword; requires standing next to a table, 1 wood, and 1 stone",
    "MAKE_IRON_SWORD": "craft an iron sword; requires standing next to a table and furnace, plus 1 wood, 1 coal, and 1 iron",
    "ASK_OPERATOR": "ask one concrete question when the next safe move is genuinely unclear",
}


def _iter_actions(actions: Any) -> list[str]:
    if actions is None:
        return []
    if isinstance(actions, str):
        return [actions.strip()] if actions.strip() else []
    if isinstance(actions, Iterable):
        return [str(action).strip() for action in actions if str(action).strip()]
    return [str(actions).strip()]


def render(actions: Any) -> str:
    lines = ["The possible list of actions you can take:"]
    for action in _iter_actions(actions):
        description = ACTION_DESCRIPTIONS.get(action, "valid environment action")
        lines.append(f"- {action}: {description}.")
    return "\n".join(lines).strip()
