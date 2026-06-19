from craftext.environment.craftext_constants import Scenarios
from craftext.environment.scenarious.checkers.achivments import checker_acvievments
from craftext.environment.scenarious.checkers.time_constrained import checker_time_placement
from craftext.environment.scenarious.checkers.building_star import checker_star
from craftext.environment.scenarious.checkers.building_line import checker_line
from craftext.environment.scenarious.checkers.building_square import checker_square
from craftext.environment.scenarious.checkers.conditional import checker_conditional_placement
from craftext.environment.scenarious.checkers.relevant import cheker_localization


CHECKER_REGISTRY = {
    Scenarios.CONDITIONAL_ACHIEVEMENTS: lambda gd, ts: checker_acvievments(gd, ts.achievements),
    Scenarios.CONDITIONAL_PLACING: lambda gd, ts: checker_conditional_placement(gd, ts.conditional_placing),
    Scenarios.LOCALIZATION_PLACE: lambda gd, ts: cheker_localization(gd, ts.Localization_placing),
    Scenarios.BUILD_LINE: lambda gd, ts: checker_line(gd, ts.building_line),
    Scenarios.BUILD_SQUARE: lambda gd, ts: checker_square(gd, ts.building_square),
    Scenarios.BUILD_STAR: lambda gd, ts: checker_star(gd, ts.building_star),
    Scenarios.TIME_CONSTRAINED_PLACEMENT: lambda gd, ts: checker_time_placement(gd, ts.time_placement),
    Scenarios.EXPLORE: lambda gd, ts: checker_acvievments(gd, ts.achievements),
}

CHECKER_FUNCTIONS = tuple(CHECKER_REGISTRY[key] for key in sorted(CHECKER_REGISTRY.keys(), key=lambda x: x.value))
