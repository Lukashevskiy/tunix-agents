"""Interactive manual play loop for one CagedCrafText scenario."""

import argparse
import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pygame
from pygame.colordict import THECOLORS as colors

from .caged_craftext_wrapper import CMDPInstructionWrapper, TextEnvStateWithConstraint
from .scenarious.loader import CagedCraftextScenariosConfigLoader
from .world_presets import build_env_and_params, build_world_preset_spec

from craftext.environment.scenarious.loader import resolve_base_environment

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_CONFIG = "budget/achievements/easy/explore_energy_above_8"
DEFAULT_WORLD_PRESET = "caged_craftext_play"

HEADER_HEIGHT = 190
SIDEBAR_WIDTH = 340

ACTION_KEY_SPECS = (
    (pygame.K_q, "NOOP"),
    (pygame.K_w, "UP"),
    (pygame.K_d, "RIGHT"),
    (pygame.K_s, "DOWN"),
    (pygame.K_a, "LEFT"),
    (pygame.K_SPACE, "DO"),
    (pygame.K_t, "PLACE_TABLE"),
    (pygame.K_TAB, "SLEEP"),
    (pygame.K_r, "PLACE_STONE"),
    (pygame.K_f, "PLACE_FURNACE"),
    (pygame.K_p, "PLACE_PLANT"),
    (pygame.K_1, "MAKE_WOOD_PICKAXE"),
    (pygame.K_2, "MAKE_STONE_PICKAXE"),
    (pygame.K_3, "MAKE_IRON_PICKAXE"),
    (pygame.K_4, "MAKE_WOOD_SWORD"),
    (pygame.K_5, "MAKE_STONE_SWORD"),
    (pygame.K_6, "MAKE_IRON_SWORD"),
    (pygame.K_7, "MAKE_DIAMOND_PICKAXE"),
    (pygame.K_8, "MAKE_DIAMOND_SWORD"),
    (pygame.K_e, "REST"),
    (pygame.K_COMMA, "ASCEND"),
    (pygame.K_PERIOD, "DESCEND"),
    (pygame.K_y, "MAKE_IRON_ARMOUR"),
    (pygame.K_u, "MAKE_DIAMOND_ARMOUR"),
    (pygame.K_i, "SHOOT_ARROW"),
    (pygame.K_o, "MAKE_ARROW"),
    (pygame.K_g, "CAST_FIREBALL"),
    (pygame.K_h, "CAST_ICEBALL"),
    (pygame.K_j, "PLACE_TORCH"),
    (pygame.K_z, "DRINK_POTION_RED"),
    (pygame.K_x, "DRINK_POTION_GREEN"),
    (pygame.K_c, "DRINK_POTION_BLUE"),
    (pygame.K_v, "DRINK_POTION_PINK"),
    (pygame.K_b, "DRINK_POTION_CYAN"),
    (pygame.K_n, "DRINK_POTION_YELLOW"),
    (pygame.K_m, "READ_BOOK"),
    (pygame.K_k, "ENCHANT_SWORD"),
    (pygame.K_l, "ENCHANT_ARMOUR"),
    (pygame.K_LEFTBRACKET, "MAKE_TORCH"),
    (pygame.K_RIGHTBRACKET, "LEVEL_UP_DEXTERITY"),
    (pygame.K_MINUS, "LEVEL_UP_STRENGTH"),
    (pygame.K_EQUALS, "LEVEL_UP_INTELLIGENCE"),
    (pygame.K_SEMICOLON, "ENCHANT_BOW"),
)

HUD_CONTROL_LINES = (
    "Move: WASD",
    "Act: SPACE",
    "Sleep: TAB",
    "Rest: E",
    "Restart: F5",
    "Quit: close window",
)


def _as_python_bool(value: Any) -> bool:
    return bool(np.asarray(value))


def _resolve_explicit_play_env_name(value: str) -> str:
    stripped_value = value.strip()
    if stripped_value in {"Craftax-Classic-Pixels-v1", "Craftax-Pixels-v1"}:
        return stripped_value

    resolved_base_env = resolve_base_environment(stripped_value)
    return "Craftax-Classic-Pixels-v1" if resolved_base_env.family == "classic" else "Craftax-Pixels-v1"


def resolve_play_env_name(config: Any, env_override: Optional[str]) -> str:
    if env_override:
        return _resolve_explicit_play_env_name(env_override)
    return _resolve_explicit_play_env_name(config.base_environment)


def load_rendering_runtime(env_name: str) -> Tuple[Any, Any, Any, Any, Any, Any]:
    resolved_base_env = resolve_base_environment(env_name)
    if resolved_base_env.family == "classic":
        from craftax.craftax_classic.constants import (
            Action,
            Achievement,
            BLOCK_PIXEL_SIZE_HUMAN,
            INVENTORY_OBS_HEIGHT,
            OBS_DIM,
        )
        from craftax.craftax_classic.renderer import render_craftax_pixels
    else:
        from craftax.craftax.constants import (
            Action,
            Achievement,
            BLOCK_PIXEL_SIZE_HUMAN,
            INVENTORY_OBS_HEIGHT,
            OBS_DIM,
        )
        from craftax.craftax.renderer import render_craftax_pixels

    return OBS_DIM, BLOCK_PIXEL_SIZE_HUMAN, INVENTORY_OBS_HEIGHT, Action, Achievement, render_craftax_pixels


def build_key_mapping(action_enum: Any) -> Dict[int, Any]:
    mapping: Dict[int, Any] = {}
    for key_code, action_name in ACTION_KEY_SPECS:
        if hasattr(action_enum, action_name):
            mapping[key_code] = getattr(action_enum, action_name)
    return mapping


def print_new_achievements(old_achievements: jax.Array, new_achievements: jax.Array, achievement_enum: Any) -> None:
    for i in range(len(old_achievements)):
        if old_achievements[i] == 0 and new_achievements[i] == 1:
            print(f"{achievement_enum(i).name} ({new_achievements.sum()}/{len(achievement_enum)})")


class CagedCrafTextRenderer:
    def __init__(
        self,
        *,
        render_fn: Any,
        obs_dim: Tuple[int, int],
        block_pixel_size_human: int,
        inventory_obs_height: int,
        scenario_handler: Any,
        pixel_render_size: int = 1,
    ) -> None:
        self.pixel_render_size = pixel_render_size
        self.pygame_events: List[Any] = []
        self.block_pixel_size_human = block_pixel_size_human
        self.scenario_handler = scenario_handler

        self.field_width = obs_dim[1] * block_pixel_size_human * pixel_render_size
        self.field_height = (obs_dim[0] + inventory_obs_height) * block_pixel_size_human * pixel_render_size
        self.screen_size = (self.field_width + SIDEBAR_WIDTH, self.field_height + HEADER_HEIGHT)

        pygame.init()
        pygame.display.set_caption("CagedCrafText")
        pygame.key.set_repeat(250, 75)
        logger.info("pygame initialized")
        self.title_font = pygame.font.Font(pygame.font.get_default_font(), 22)
        self.subtitle_font = pygame.font.Font(pygame.font.get_default_font(), 18)
        self.body_font = pygame.font.Font(pygame.font.get_default_font(), 16)
        self.small_font = pygame.font.Font(pygame.font.get_default_font(), 14)
        self.screen_surface = pygame.display.set_mode(self.screen_size)
        self._render = jax.jit(render_fn, static_argnums=(1,))
        logger.info("init render succesfull")
    def update(self) -> None:
        self.pygame_events = list(pygame.event.get())

    def render_field(self, env_state: TextEnvStateWithConstraint[Any]) -> jax.Array:
        pixels = self._render(env_state.env_state, block_pixel_size=self.block_pixel_size_human)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=0)
        pixels = jnp.repeat(pixels, repeats=self.pixel_render_size, axis=1)
        return pixels

    def is_quit_requested(self) -> bool:
        return any(event.type == pygame.QUIT for event in self.pygame_events)

    def is_restart_requested(self) -> bool:
        return any(event.type == pygame.KEYDOWN and event.key == pygame.K_F5 for event in self.pygame_events)

    def get_action_from_keypress(self, state: Any, key_mapping: Dict[int, Any]) -> Optional[int]:
        if _as_python_bool(getattr(state, "is_sleeping", False)) or _as_python_bool(getattr(state, "is_resting", False)):
            noop = key_mapping.get(pygame.K_q)
            return None if noop is None else noop.value

        for event in self.pygame_events:
            if event.type == pygame.KEYDOWN and event.key in key_mapping:
                return key_mapping[event.key].value
        return None

    def render(self, *, env_state: TextEnvStateWithConstraint[Any], last_reward: float) -> None:
        self.screen_surface.fill((12, 14, 18))
        self._draw_header(env_state=env_state, last_reward=last_reward)
        self._draw_field(env_state)
        self._draw_sidebar(env_state=env_state, last_reward=last_reward)
        pygame.display.flip()

    def _draw_header(self, *, env_state: TextEnvStateWithConstraint[Any], last_reward: float) -> None:
        header_rect = pygame.Rect(0, 0, self.screen_size[0], HEADER_HEIGHT)
        pygame.draw.rect(self.screen_surface, (230, 235, 241), header_rect)
        pygame.draw.line(self.screen_surface, (180, 188, 198), (0, HEADER_HEIGHT - 1), (self.screen_size[0], HEADER_HEIGHT - 1), 1)

        instruction = self.scenario_handler.scenario_data.instructions_list[int(env_state.idx)]
        constraint = self.scenario_handler.scenario_data.texutal_constraints_list[int(env_state.idx)]
        title = self.title_font.render(instruction, True, (18, 24, 33))
        self.screen_surface.blit(title, (20, 18))

        meta_lines = [
            f"Constraint: {constraint or '(none)'}",
            f"Reward: {last_reward:.2f} | Step cost: {float(np.asarray(env_state.cost)):.1f} | Episode cost: {float(np.asarray(env_state.episode_cost)):.1f}",
        ]
        cursor_y = 56
        for line in meta_lines:
            cursor_y = self._draw_wrapped_line(line, 20, cursor_y, self.field_width + SIDEBAR_WIDTH - 40, self.subtitle_font, (42, 55, 70))

    def _draw_field(self, env_state: TextEnvStateWithConstraint[Any]) -> None:
        image = np.array(self.render_field(env_state))
        surface = pygame.surfarray.make_surface(image.transpose((1, 0, 2)))
        self.screen_surface.blit(surface, (0, HEADER_HEIGHT))

    def _draw_sidebar(self, *, env_state: TextEnvStateWithConstraint[Any], last_reward: float) -> None:
        sidebar_rect = pygame.Rect(self.field_width, HEADER_HEIGHT, SIDEBAR_WIDTH, self.field_height)
        pygame.draw.rect(self.screen_surface, (26, 31, 38), sidebar_rect)
        pygame.draw.line(self.screen_surface, (64, 74, 86), (self.field_width, HEADER_HEIGHT), (self.field_width, self.screen_size[1]), 1)

        state = env_state.env_state
        achievements_done = int(np.asarray(state.achievements).sum()) if hasattr(state, "achievements") else 0
        achievement_total = len(np.asarray(state.achievements)) if hasattr(state, "achievements") else 0
        stats = [
            ("Energy", getattr(state, "player_energy", None)),
            ("Food", getattr(state, "player_food", None)),
            ("Drink", getattr(state, "player_drink", None)),
            ("Health", getattr(state, "player_health", None)),
            ("Fatigue", getattr(state, "player_fatigue", None)),
            ("Recover", getattr(state, "player_recover", None)),
            ("Timestep", getattr(env_state, "timestep", None)),
            ("Reward", last_reward),
            ("Step cost", float(np.asarray(env_state.cost))),
            ("Episode cost", float(np.asarray(env_state.episode_cost))),
            ("Achievements", f"{achievements_done}/{achievement_total}" if achievement_total else "-"),
            ("Sleeping", "yes" if _as_python_bool(getattr(state, "is_sleeping", False)) else "no"),
            ("Resting", "yes" if _as_python_bool(getattr(state, "is_resting", False)) else "no"),
        ]

        title = self.title_font.render("HUD", True, colors["white"])
        self.screen_surface.blit(title, (self.field_width + 18, HEADER_HEIGHT + 16))

        y = HEADER_HEIGHT + 56
        for label, raw_value in stats:
            text = self._format_stat_value(raw_value)
            label_surface = self.small_font.render(label.upper(), True, (140, 152, 168))
            value_surface = self.body_font.render(text, True, colors["white"])
            self.screen_surface.blit(label_surface, (self.field_width + 18, y))
            self.screen_surface.blit(value_surface, (self.field_width + 18, y + 16))
            y += 42

        controls_y = self.screen_size[1] - (len(HUD_CONTROL_LINES) * self.small_font.get_linesize()) - 24
        controls_title = self.title_font.render("Controls", True, colors["white"])
        self.screen_surface.blit(controls_title, (self.field_width + 18, controls_y - 30))
        for line in HUD_CONTROL_LINES:
            surface = self.small_font.render(line, True, (210, 219, 228))
            self.screen_surface.blit(surface, (self.field_width + 18, controls_y))
            controls_y += self.small_font.get_linesize()

    def _format_stat_value(self, raw_value: Any) -> str:
        if raw_value is None:
            return "-"
        if isinstance(raw_value, str):
            return raw_value
        array_value = np.asarray(raw_value)
        if array_value.ndim == 0:
            scalar = array_value.item()
            if isinstance(scalar, (float, np.floating)):
                return f"{float(scalar):.2f}"
            return str(int(scalar)) if isinstance(scalar, (int, np.integer, bool, np.bool_)) else str(scalar)
        return str(raw_value)

    def _draw_wrapped_line(self, text: str, x: int, y: int, max_width: int, font: Any, color: Any) -> int:
        words = text.split()
        if not words:
            return y + font.get_linesize()
        current = words[0]
        cursor_y = y
        for word in words[1:]:
            candidate = f"{current} {word}"
            if font.size(candidate)[0] <= max_width:
                current = candidate
            else:
                self.screen_surface.blit(font.render(current, True, color), (x, cursor_y))
                cursor_y += font.get_linesize()
                current = word
        self.screen_surface.blit(font.render(current, True, color), (x, cursor_y))
        return cursor_y + font.get_linesize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run interactive CagedCrafText manual play.")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Caged scenario config name under CagedCrafText/dataset/configs.")
    parser.add_argument("--env", default=None, help="Explicit Craftax env id or alias. Example: Craftax-Pixels-v1, classic, full.")
    parser.add_argument("--world-preset", default=DEFAULT_WORLD_PRESET, help="World preset YAML name under CagedCrafText/world_presets.")
    parser.add_argument("--seed", type=int, default=None, help="Seed for world generation. Fixed presets derive stable noise angles from it.")
    parser.add_argument("--map-size", type=int, default=None, help="Override square world size for presets that rebuild the underlying env.")
    parser.add_argument("--box-inner-size", type=int, default=None, help="Override playable square size for box presets.")
    parser.add_argument("--perimeter-tree-prob", type=float, default=None, help="Tree probability on the one-tile perimeter for box presets.")
    parser.add_argument("--instruction-idx", type=int, default=0, help="Fixed instruction index to run.")
    parser.add_argument("--pixel-render-size", type=int, default=1, help="Integer upscaling factor for the rendered game view.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CagedCraftextScenariosConfigLoader.load_config(args.config)
    env_name = resolve_play_env_name(config=config, env_override=args.env)

    map_section: Dict[str, Any] = {}
    if args.box_inner_size is not None or args.perimeter_tree_prob is not None:
        map_section["generator"] = {
            "name": "box",
            "config": {
                "inner_size": 5 if args.box_inner_size is None else args.box_inner_size,
                "perimeter_tree_prob": 1.0 if args.perimeter_tree_prob is None else args.perimeter_tree_prob,
            },
        }

    world_preset_spec = build_world_preset_spec(
        env_name=env_name,
        preset_name=args.world_preset,
        seed=args.seed,
        systems={"static_env": {"map_size": args.map_size}},
        map=map_section,
    )

    obs_dim, block_pixel_size_human, inventory_obs_height, action_enum, achievement_enum, render_fn = load_rendering_runtime(env_name)
    key_mapping = build_key_mapping(action_enum)
    env, env_params = build_env_and_params(world_preset_spec, auto_reset=False)
    wrapper = CMDPInstructionWrapper(env, config_name=args.config)

    renderer = CagedCrafTextRenderer(
        render_fn=render_fn,
        obs_dim=obs_dim,
        block_pixel_size_human=block_pixel_size_human,
        inventory_obs_height=inventory_obs_height,
        scenario_handler=wrapper.scenario_handler,
        pixel_render_size=args.pixel_render_size,
    )

    logger.info("Play config: %s", args.config)
    logger.info("Resolved env: %s", env_name)
    logger.info("World preset: %s", world_preset_spec.name)
    logger.info("World seed: %s", world_preset_spec.env.seed)
    logger.info("Instruction row: %s", args.instruction_idx)

    rng = jax.random.PRNGKey(world_preset_spec.env.seed)
    rng, reset_rng = jax.random.split(rng)
    _, env_state = wrapper.reset(reset_rng, env_params, instruction_idx=args.instruction_idx)
    last_reward = 0.0

    print("Controls")
    for key_code, action in key_mapping.items():
        print(f"{pygame.key.name(key_code)}: {action.name.lower()}")
    print("F5: restart\n")

    clock = pygame.time.Clock()

    renderer.render(env_state=env_state, last_reward=0)
    while True:
        renderer.update()
        if renderer.is_quit_requested():
            return
        if renderer.is_restart_requested():
            rng, reset_rng = jax.random.split(rng)
            _, env_state = wrapper.reset(reset_rng, env_params, instruction_idx=args.instruction_idx)
            last_reward = 0.0

        action = renderer.get_action_from_keypress(env_state.env_state, key_mapping)
        if action is not None:
            rng, step_rng, reset_rng = jax.random.split(rng, 3)
            old_achievements = env_state.env_state.achievements
            _, env_state, reward, done, _ = wrapper.step(step_rng, env_state, action, env_params)
            last_reward = float(np.asarray(reward))
            print_new_achievements(old_achievements, env_state.env_state.achievements, achievement_enum)

            energy_value = getattr(env_state.env_state, "player_energy", None)
            if energy_value is not None:
                print(
                    "Energy:",
                    int(np.asarray(energy_value)),
                    "| Step cost:",
                    float(np.asarray(env_state.cost)),
                    "| Episode cost:",
                    float(np.asarray(env_state.episode_cost)),
                )
            if last_reward > 0.01 or last_reward < -0.01:
                print(f"Reward: {last_reward}\n")

            if bool(np.asarray(done)):
                _, env_state = wrapper.reset(reset_rng, env_params, instruction_idx=args.instruction_idx)
                last_reward = 0.0

        renderer.render(env_state=env_state, last_reward=last_reward)
        clock.tick(60)


if __name__ == "__main__":
    main()
