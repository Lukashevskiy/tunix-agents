"""Aggregated localization scenario subsets."""

from craftext.environment.scenarious.scenario_types import ScenarioMap

from ..localization_place.instructions import easy as localization_easy
from ..localization_place.instructions import medium as localization_medium

easy: ScenarioMap = localization_easy
medium: ScenarioMap = localization_medium
