from craftext.environment.scenarious.checkers.target_state import Achievements, TargetState
from craftext.environment.craftext_constants import Achievement, Scenarios, AchievementState


def create_target_state(required=None, forbidden=None):
    if required is None:
        required = []
    if forbidden is None:
        forbidden = []
    base_vector = [AchievementState.NOT_MATTER.value for _ in range(Achievement.MAKE_IRON_SWORD.value + 1)]
    for i in range(len(base_vector)):
        if i in required:
            base_vector[i] = AchievementState.NEED_TO_ACHIEVE.value
        elif i in forbidden:
            base_vector[i] = AchievementState.AVOID_TO_ACHIEVE.value
    target_achievements = Achievements(achievement_mask=tuple(base_vector))
    return TargetState(achievements=target_achievements)


one = {
    "COLLECT_WOOD": {
        "instruction": "Collect wood.",
        "instruction_paraphrases": [],
        "scenario_checker": Scenarios.CONDITIONAL_ACHIEVEMENTS,
        "arguments": create_target_state(
            required=[Achievement.COLLECT_WOOD.value],
            forbidden=[]
        )
    }
}

# Optional placeholders for consistency with other modules
easy = {}
medium = {}
hard = {}
