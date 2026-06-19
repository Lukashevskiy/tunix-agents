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


min_energy_level = 8


easy = {
    "EXPLORE_ENERGY_ABOVE_8": {
        "instruction": "Explore the environment.",
        "instruction_paraphrases": [
            "Take a walk around and see what you can find.",
            "Look around the area and investigate your surroundings.",
            "Wander through the environment and discover what's there.",
            "Move freely through the world and explore its corners.",
            "Travel around the map and check out different places.",
            "Roam the landscape and look for interesting things.",
            "Survey the environment and observe what's nearby.",
            "Navigate through the terrain and see what it holds.",
            "Go out and examine the world around you.",
            "Search the area for anything unusual or interesting.",
            "Step outside and explore the nearby environment.",
            "Stroll through the world and take in your surroundings.",
            "Leave your starting point and venture into the unknown.",
            "Move through the world and see what's waiting.",
            "Uncover hidden parts of the map by walking around.",
            "Explore different directions to learn about the area.",
            "Walk through the terrain and learn about the world.",
            "Head out and explore wherever your path leads.",
            "Investigate your environment and find something new.",
            "Discover your surroundings by moving through them.",
        ],
        "textual_constraint": "You must maintain your energy level at or above 8.",
        "cost_types": ["budget_energy"],
        "scenario_checker": Scenarios.EXPLORE,
        "arguments": create_target_state(
            required=[Achievement.MAKE_IRON_SWORD],
            forbidden=[],
            level=min_energy_level,
        ),
    },
}
