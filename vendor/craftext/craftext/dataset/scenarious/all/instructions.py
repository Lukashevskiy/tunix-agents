"""Aggregated scenario subsets combining building and non-building tasks."""

from craftext.environment.scenarious.scenario_types import ScenarioMap

from ..all_building.instructions import easy as building_easy, medium as building_medium
from ..conditional_achievements.instructions import easy as cond_ach_easy, medium as cond_ach_medium
from ..conditional_placing.instructions import easy as cond_place_easy, medium as cond_place_medium
from ..explore.instructions import easy as explore_easy, medium as explore_medium
from ..localization_place.instructions import easy as loc_easy, medium as loc_medium


def _merge_with_prefix(prefix: str, data: ScenarioMap) -> ScenarioMap:
    return {f"{prefix}__{k}": v for k, v in data.items()}


def _merge_all(*parts: tuple[str, ScenarioMap]) -> ScenarioMap:
    merged: ScenarioMap = {}
    for prefix, chunk in parts:
        merged.update(_merge_with_prefix(prefix, chunk))
    return merged


non_building_easy: ScenarioMap = _merge_all(
    ("conditional_achievements", cond_ach_easy),
    ("conditional_placing", cond_place_easy),
    ("explore", explore_easy),
    ("localization_place", loc_easy),
)

non_building_medium: ScenarioMap = _merge_all(
    ("conditional_achievements", cond_ach_medium),
    ("conditional_placing", cond_place_medium),
    ("explore", explore_medium),
    ("localization_place", loc_medium),
)


easy_only_building: ScenarioMap = dict(building_easy)
easy_without_building: ScenarioMap = dict(non_building_easy)

# Legacy canonical train splits.
easy: ScenarioMap = _merge_all(("building", building_easy), ("non_building", non_building_easy))
medium: ScenarioMap = _merge_all(("building", building_medium), ("non_building", non_building_medium))
