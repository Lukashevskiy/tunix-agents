"""Aggregated building-only scenario subsets used by legacy config keys."""

from craftext.environment.scenarious.scenario_types import ScenarioMap

from ..building.line.instructions import easy as line_easy, medium as line_medium
from ..building.square.instructions import easy as square_easy, medium as square_medium
from ..building.star.instructions import easy as star_easy, medium as star_medium


def _merge_with_prefix(prefix: str, data: ScenarioMap) -> ScenarioMap:
    return {f"{prefix}__{k}": v for k, v in data.items()}


def _merge_building(*parts: tuple[str, ScenarioMap]) -> ScenarioMap:
    merged: ScenarioMap = {}
    for prefix, chunk in parts:
        merged.update(_merge_with_prefix(prefix, chunk))
    return merged


easy: ScenarioMap = _merge_building(
    ("line", line_easy),
    ("square", square_easy),
    ("star", star_easy),
)

medium: ScenarioMap = _merge_building(
    ("line", line_medium),
    ("square", square_medium),
    ("star", star_medium),
)
