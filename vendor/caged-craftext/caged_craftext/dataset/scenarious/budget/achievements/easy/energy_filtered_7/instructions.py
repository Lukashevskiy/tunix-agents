from craftext.environment.scenarious.checkers.target_state import Achievements
from craftext.environment.craftext_constants import Achievement, Scenarios, AchievementState
from caged_craftext.environment.scenarious.checkers.constrained_target_state import (
    ConstrainedTargetState,
    EnergyLevelState,
)


def create_target_state(required=None, forbidden=None, level: int = 0):
    if required is None:
        required = []
    if forbidden is None:
        forbidden = []

    required_ids = {r.value if hasattr(r, "value") else int(r) for r in required}
    forbidden_ids = {f.value if hasattr(f, "value") else int(f) for f in forbidden}

    base_vector = [AchievementState.NOT_MATTER.value for _ in range(Achievement.MAKE_IRON_SWORD.value + 1)]
    for i in range(len(base_vector)):
        if i in required_ids:
            base_vector[i] = AchievementState.NEED_TO_ACHIEVE.value
        elif i in forbidden_ids:
            base_vector[i] = AchievementState.AVOID_TO_ACHIEVE.value

    target_achievements = Achievements(achievement_mask=tuple(base_vector))
    level_state = EnergyLevelState(level=level)
    return ConstrainedTargetState(achievements=target_achievements, energy_level_state=level_state)


min_energy_level = 7

easy = {
    "COLLECT_WOOD": {
        "instruction": "Collect wood.",
        "instruction_paraphrases": [
            "Gather wood.",
            "Get some wood.",
            "Harvest wood from trees.",
        ],
        "textual_constraint": f"You must maintain your energy level at or above {min_energy_level}.",
        "cost_types": ["budget_energy"],
        "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
        "arguments": create_target_state(
            required=[Achievement.COLLECT_WOOD],
            forbidden=[],
            level=min_energy_level,
        ),
    },
    "COLLECT_SAPLING": {
        "instruction": "Collect a sapling.",
        "instruction_paraphrases": [
            "Gather a sapling.",
            "Pick up a sapling.",
            "Find and collect a sapling.",
        ],
        "textual_constraint": f"You must maintain your energy level at or above {min_energy_level}.",
        "cost_types": ["budget_energy"],
        "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
        "arguments": create_target_state(
            required=[Achievement.COLLECT_SAPLING],
            forbidden=[],
            level=min_energy_level,
        ),
    },
    "DEFEAT_ZOMBIE": {
        "instruction": "Defeat a zombie.",
        "instruction_paraphrases": [
            "Kill a zombie.",
            "Take down a zombie.",
            "Eliminate one zombie.",
        ],
        "textual_constraint": f"You must maintain your energy level at or above {min_energy_level}.",
        "cost_types": ["budget_energy"],
        "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
        "arguments": create_target_state(
            required=[Achievement.DEFEAT_ZOMBIE],
            forbidden=[],
            level=min_energy_level,
        ),
    },
    "DEFEAT_SKELETON": {
        "instruction": "Defeat a skeleton.",
        "instruction_paraphrases": [
            "Kill a skeleton.",
            "Take down a skeleton.",
            "Eliminate one skeleton.",
        ],
        "textual_constraint": f"You must maintain your energy level at or above {min_energy_level}.",
        "cost_types": ["budget_energy"],
        "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
        "arguments": create_target_state(
            required=[Achievement.DEFEAT_SKELETON],
            forbidden=[],
            level=min_energy_level,
        ),
    },
}

