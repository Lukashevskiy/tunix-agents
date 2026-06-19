"""Aggregated conditional scenario subsets."""

from craftext.environment.scenarious.scenario_types import ScenarioMap

from ..conditional_achievements.instructions import easy as cond_ach_easy, medium as cond_ach_medium
from ..conditional_placing.instructions import easy as cond_place_easy, medium as cond_place_medium


def _merge_with_prefix(prefix: str, data: ScenarioMap) -> ScenarioMap:
    return {f"{prefix}__{k}": v for k, v in data.items()}


def _merge_conditional(*parts: tuple[str, ScenarioMap]) -> ScenarioMap:
    merged: ScenarioMap = {}
    for prefix, chunk in parts:
        merged.update(_merge_with_prefix(prefix, chunk))
    return merged


easy: ScenarioMap = _merge_conditional(
    ("achievements", cond_ach_easy),
    ("placing", cond_place_easy),
)

medium: ScenarioMap = _merge_conditional(
    ("achievements", cond_ach_medium),
    ("placing", cond_place_medium),
)
