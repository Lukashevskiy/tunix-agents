"""Typed contracts for scenario entries loaded from dataset modules."""

from typing import Dict, List, TypedDict

from craftext.environment.craftext_constants import Scenarios
from craftext.environment.scenarious.checkers.target_state import TargetState


class ScenarioEntryRequired(TypedDict):
    """Required fields for one scenario entry."""

    instruction: str
    scenario_checker: Scenarios
    arguments: TargetState


class ScenarioEntry(ScenarioEntryRequired, total=False):
    """Optional fields for one scenario entry."""

    instruction_paraphrases: List[str]
    str_check_lambda: str


ScenarioMap = Dict[str, ScenarioEntry]
