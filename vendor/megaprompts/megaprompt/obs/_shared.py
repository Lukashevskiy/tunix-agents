from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from craftax.craftax_classic.constants import BlockType
except Exception:  # pragma: no cover
    BlockType = None  # type: ignore


_DIR_NAMES = {1: "up", 2: "right", 3: "down", 4: "left"}
_DIR_DELTAS = {1: (-1, 0), 2: (0, 1), 3: (1, 0), 4: (0, -1)}

_ASCII_MAPPING = {
    "out_of_bounds": "#",
    "unknown": "?",
    "water": "~",
    "grass": ".",
    "stone": "%",
    "tree": "T",
    "wood": "w",
    "sand": ":",
    "lava": "L",
    "coal": "c",
    "iron": "i",
    "diamond": "d",
    "gold": "g",
    "path": "_",
    "table": "T",
    "crafting_table": "T",
    "furnace": "F",
    "plant": "*",
    "ripe_plant": "P",
    "bed": "B",
}


@dataclass(frozen=True)
class SymbolLegend:
    symbol: str
    name: str


def _tile_name_from_id(block_id: int) -> str:
    if BlockType is None:
        return "Unknown"
    try:
        bt = BlockType(int(block_id))
    except Exception:
        return "Unknown"
    return bt.name.title().replace("_", " ")


def _tile_name_lower_from_id(block_id: int) -> str:
    if BlockType is None:
        return "unknown"
    try:
        bt = BlockType(int(block_id))
    except Exception:
        return "unknown"
    return bt.name.lower()


def _iter_non_grass_tiles(state: Any) -> List[Tuple[str, Tuple[int, int], int]]:
    map_array = np.asarray(state.map)
    h, w = map_array.shape
    tiles: List[Tuple[str, Tuple[int, int], int]] = []

    if BlockType is not None:
        oob = int(BlockType.OUT_OF_BOUNDS.value)
        grass = int(getattr(BlockType, "GRASS").value)
        darkness = int(getattr(BlockType, "DARKNESS", BlockType.OUT_OF_BOUNDS).value)
    else:
        oob, grass, darkness = -999999, -999998, -999997

    for x in range(h):
        for y in range(w):
            block_id = int(map_array[x, y])
            if block_id in (oob, grass, darkness):
                continue
            if block_id < 0:
                continue
            tiles.append((_tile_name_from_id(block_id), (x, y), block_id))
    return tiles


def _nearest_unique_tiles(state: Any, k: int = 5) -> List[Tuple[str, Tuple[int, int]]]:
    px = int(state.player_position[0])
    py = int(state.player_position[1])
    tiles = _iter_non_grass_tiles(state)
    tiles_sorted = sorted(tiles, key=lambda t: abs(px - t[1][0]) + abs(py - t[1][1]))

    seen: set[str] = set()
    out: List[Tuple[str, Tuple[int, int]]] = []
    for name, pos, _block_id in tiles_sorted:
        if name in seen:
            continue
        seen.add(name)
        out.append((name, pos))
        if len(out) >= k:
            break
    return out


def _front_block(state: Any) -> Tuple[str, Optional[Tuple[int, int]]]:
    px = int(state.player_position[0])
    py = int(state.player_position[1])
    dir_idx = int(getattr(state, "player_direction", 0))
    if dir_idx not in _DIR_DELTAS:
        return "Unknown", None

    dx, dy = _DIR_DELTAS[dir_idx]
    fx, fy = px + int(dx), py + int(dy)
    map_array = np.asarray(state.map)
    h, w = map_array.shape
    if not (0 <= fx < h and 0 <= fy < w):
        return "Out of bounds", (fx, fy)
    block_id = int(map_array[fx, fy])
    return _tile_name_from_id(block_id), (fx, fy)


def _current_block(state: Any) -> Tuple[str, Tuple[int, int]]:
    px = int(state.player_position[0])
    py = int(state.player_position[1])
    map_array = np.asarray(state.map)
    h, w = map_array.shape
    if 0 <= px < h and 0 <= py < w:
        block_id = int(map_array[px, py])
        return _tile_name_from_id(block_id), (px, py)
    return "Out of bounds", (px, py)


def _moves_to_reach(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
    fx, fy = from_pos
    tx, ty = to_pos
    dx = tx - fx
    dy = ty - fy

    parts: List[str] = []
    if dx < 0:
        parts.append("upward")
    elif dx > 0:
        parts.append("downward")
    if dy < 0:
        parts.append("left")
    elif dy > 0:
        parts.append("right")

    return " and ".join(parts) if parts else "stay here (already there)"


def _render_inventory(state: Any) -> str:
    inv = getattr(state, "inventory", None)
    if inv is None:
        return "Inventory: unavailable"

    items: List[str] = []

    if isinstance(inv, Mapping):
        for name, val in inv.items():
            try:
                if hasattr(val, "item"):
                    val = val.item()
                val_int = int(val)
            except Exception:
                continue
            if val_int > 0:
                items.append(f"{name}={val_int}")
    else:
        fields = getattr(inv.__class__, "__dataclass_fields__", None)
        if fields:
            for name in fields.keys():
                try:
                    val = getattr(inv, name)
                except Exception:
                    continue
                try:
                    if hasattr(val, "item"):
                        val = val.item()
                    val_int = int(val)
                except Exception:
                    continue
                if val_int > 0:
                    items.append(f"{name}={val_int}")
        else:
            for name in dir(inv):
                if name.startswith("_"):
                    continue
                try:
                    val = getattr(inv, name)
                except Exception:
                    continue
                if callable(val):
                    continue
                try:
                    if hasattr(val, "item"):
                        val = val.item()
                    val_int = int(val)
                except Exception:
                    continue
                if val_int > 0:
                    items.append(f"{name}={val_int}")

    if not items:
        return "Inventory: empty"
    return "Inventory: " + ", ".join(sorted(items))


def render_map(state: Any) -> Tuple[str, List[SymbolLegend]]:
    map_array = np.asarray(state.map)
    h, w = map_array.shape
    px = int(state.player_position[0])
    py = int(state.player_position[1])

    size = 10
    half = size // 2
    top_x = px - half
    left_y = py - half

    header = "y\\x " + " ".join(f"{(left_y + j):2d}" for j in range(size))
    rows: List[str] = [header]
    seen_symbols: Dict[str, str] = {}
    legend: List[SymbolLegend] = [SymbolLegend("@", "You")]

    for i in range(size):
        row_x = top_x + i
        row_cells: List[str] = []
        for j in range(size):
            col_y = left_y + j
            if row_x == px and col_y == py:
                row_cells.append("@")
                continue
            if not (0 <= row_x < h and 0 <= col_y < w):
                row_cells.append("#")
                seen_symbols["#"] = "out_of_bounds"
                continue

            block_id = int(map_array[row_x, col_y])
            name = _tile_name_lower_from_id(block_id)
            sym = _ASCII_MAPPING.get(name, (name[0].upper() if name else "?"))
            row_cells.append(sym)
            if sym not in ("@",) and sym not in seen_symbols:
                seen_symbols[sym] = name

        rows.append(f"{row_x:3d} " + "  ".join(row_cells))

    for sym in sorted(seen_symbols.keys()):
        if sym == "#":
            legend.append(SymbolLegend(sym, "out_of_bounds"))
        else:
            legend.append(SymbolLegend(sym, seen_symbols[sym]))

    return "\n".join(rows), legend


def render_coords_text(state: Any, k_nearest: int = 5) -> str:
    px = int(state.player_position[0])
    py = int(state.player_position[1])
    dir_idx = int(getattr(state, "player_direction", 0))
    facing = _DIR_NAMES.get(dir_idx, "unknown")
    current_name, (cx, cy) = _current_block(state)
    front_name, front_pos = _front_block(state)

    lines: List[str] = []
    lines.append(f"You are at coord y={px}, x={py}. You are rotated {facing}.")
    lines.append(f"You are standing on {current_name} at y={cx}, x={cy}.")
    if front_pos is not None:
        fx, fy = front_pos
        lines.append(f"In front of you there is {front_name} at y={fx}, x={fy}.")
    else:
        lines.append(f"In front of you there is {front_name}.")

    lines.append("Nearby objects you can see:")
    for name, (x, y) in _nearest_unique_tiles(state, k=k_nearest):
        plan = _moves_to_reach((px, py), (x, y))
        lines.append(f"- You can see {name} at y={x}, x={y}. To reach it, move {plan}.")
    return "\n".join(lines).strip()


def render_map_text(state: Any, include_inventory: bool = True) -> str:
    map_text, legend = render_map(state)
    legend_text = ", ".join(f"'{item.symbol}': {item.name}" for item in legend)
    inventory_block = f"\n\n{_render_inventory(state)}" if include_inventory else ""
    return (
        "Symbolic map 10x10 (agent in the middle). Coordinates shown as [y, x].\n"
        f"{map_text}"
        f"{inventory_block}\n\n"
        f"Legend: {legend_text}"
    ).strip()


def render_map_and_coords_text(state: Any) -> str:
    return f"{render_coords_text(state)}\n\n{render_map_text(state, include_inventory=True)}".strip()
