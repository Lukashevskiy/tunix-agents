#!/usr/bin/env python3
"""Visualize a saved tunix-craftext replay trajectory with pygame.

This script shows every step field plus the observation image if the saved
trajectory contains a numeric array observation.
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pygame


def load_trajectory(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "steps" not in payload:
        raise ValueError(f"Trajectory file {path} is not a replay artifact")
    if not isinstance(payload["steps"], list):
        raise ValueError(f"Trajectory file {path} contains invalid steps")
    return payload


def normalize_image(obs: Any) -> np.ndarray | None:
    try:
        array = np.asarray(obs)
    except Exception:
        return None
    if array.size == 0:
        return None
    if array.ndim == 1:
        return None
    if array.ndim == 2:
        image = array
        image = image.astype(np.float32)
        image = image - image.min()
        if image.max() > 0:
            image = image / image.max() * 255.0
        image = np.stack([image, image, image], axis=-1)
    elif array.ndim == 3 and array.shape[2] in {1, 3, 4}:
        image = array.astype(np.float32)
        if image.dtype != np.uint8:
            image = image - image.min()
            if image.max() > 0:
                image = image / image.max() * 255.0
            image = image.astype(np.uint8)
        if image.shape[2] == 1:
            image = np.concatenate([image, image, image], axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
    else:
        return None
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def render_multiline_text(
    surface: pygame.Surface,
    text: str,
    font: pygame.font.Font,
    color: tuple[int, int, int],
    rect: pygame.Rect,
    line_spacing: int = 2,
) -> None:
    lines = textwrap.wrap(text, width=70)
    x, y = rect.topleft
    for line in lines:
        rendered = font.render(line, True, color)
        surface.blit(rendered, (x, y))
        y += rendered.get_height() + line_spacing


def format_value(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float, str)):
        return str(value)
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return "[]"
        return f"[{type(value[0]).__name__} x {len(value)}]"
    if isinstance(value, dict):
        return f"{{keys={len(value)}}}"
    return type(value).__name__


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=Path("artifacts/trajectories/qwen-craftext-latest.json"),
        help="Path to the saved replay JSON trajectory.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1400,
        help="Window width in pixels.",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=960,
        help="Window height in pixels.",
    )
    args = parser.parse_args()

    payload = load_trajectory(args.trajectory)
    steps = payload["steps"]
    if not steps:
        raise ValueError("Trajectory contains no steps")

    pygame.init()
    screen = pygame.display.set_mode((args.window_width, args.window_height))
    pygame.display.set_caption("Tunix-CrafText Trajectory Viewer")
    clock = pygame.time.Clock()

    title_font = pygame.font.SysFont(None, 28)
    header_font = pygame.font.SysFont(None, 24)
    body_font = pygame.font.SysFont(None, 18)

    step_index = 0
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return
            if event.type == pygame.KEYDOWN:
                if event.key in {pygame.K_q, pygame.K_ESCAPE}:
                    return
                if event.key == pygame.K_RIGHT:
                    step_index = min(step_index + 1, len(steps) - 1)
                if event.key == pygame.K_LEFT:
                    step_index = max(step_index - 1, 0)
                if event.key == pygame.K_HOME:
                    step_index = 0
                if event.key == pygame.K_END:
                    step_index = len(steps) - 1

        screen.fill((18, 18, 22))

        header_text = f"Trajectory: {args.trajectory.name}    Step: {step_index + 1}/{len(steps)}"
        header_surface = title_font.render(header_text, True, (240, 240, 240))
        screen.blit(header_surface, (20, 20))

        step = steps[step_index]
        left_rect = pygame.Rect(20, 60, 620, args.window_height - 80)
        right_rect = pygame.Rect(660, 60, args.window_width - 680, args.window_height - 80)

        pygame.draw.rect(screen, (34, 34, 40), left_rect)
        pygame.draw.rect(screen, (34, 34, 40), right_rect)
        pygame.draw.rect(screen, (64, 64, 74), left_rect, 2)
        pygame.draw.rect(screen, (64, 64, 74), right_rect, 2)

        metadata = [
            ("action_id", step.get("action_id")),
            ("action_label", step.get("action_label")),
            ("reward", step.get("reward")),
            ("terminated", step.get("terminated")),
            ("truncated", step.get("truncated")),
            ("fallback_used", step.get("fallback_used")),
            ("invalid_format", step.get("invalid_format")),
            ("unknown_action", step.get("unknown_action")),
            ("token_ids", format_value(step.get("token_ids"))),
            ("prompt_token_ids", format_value(step.get("prompt_token_ids"))),
            ("token_logprobs", format_value(step.get("token_logprobs"))),
            ("action_mask", format_value(step.get("action_mask"))),
            ("observation", format_value(step.get("observation"))),
        ]

        y = left_rect.top + 12
        header = header_font.render("Step metadata", True, (220, 220, 220))
        screen.blit(header, (left_rect.left + 12, y))
        y += header.get_height() + 10
        for label, value in metadata:
            line = f"{label}: {value}"
            rendered = body_font.render(line, True, (210, 210, 210))
            screen.blit(rendered, (left_rect.left + 12, y))
            y += rendered.get_height() + 6
            if y > left_rect.bottom - 120:
                break

        text_x = left_rect.left + 12
        text_y = left_rect.top + 260
        prompt_header = header_font.render("Prompt", True, (220, 220, 220))
        screen.blit(prompt_header, (text_x, text_y))
        text_y += prompt_header.get_height() + 8
        render_multiline_text(screen, str(step.get("prompt", "")), body_font, (240, 240, 240), pygame.Rect(text_x, text_y, left_rect.width - 24, 240))

        completion_y = left_rect.top + 520
        completion_header = header_font.render("Completion", True, (220, 220, 220))
        screen.blit(completion_header, (text_x, completion_y))
        completion_y += completion_header.get_height() + 8
        render_multiline_text(screen, str(step.get("raw_completion", "")), body_font, (240, 240, 240), pygame.Rect(text_x, completion_y, left_rect.width - 24, 340))

        obs = step.get("observation")
        obs_image = normalize_image(obs)
        if obs_image is not None:
            surface = pygame.surfarray.make_surface(obs_image.transpose((1, 0, 2)))
            image_rect = surface.get_rect()
            image_rect.center = right_rect.center
            max_width = right_rect.width - 24
            max_height = right_rect.height - 24
            scale = min(max_width / image_rect.width, max_height / image_rect.height, 1.0)
            if scale != 1.0:
                new_size = (max(1, int(image_rect.width * scale)), max(1, int(image_rect.height * scale)))
                surface = pygame.transform.smoothscale(surface, new_size)
                image_rect = surface.get_rect(center=right_rect.center)
            screen.blit(surface, image_rect)
            note = body_font.render("Observation image", True, (210, 210, 210))
            screen.blit(note, (right_rect.left + 12, right_rect.bottom - 28))
        else:
            missing_text = "Observation is not renderable as an image.\nShow raw summary here."
            render_multiline_text(screen, missing_text, body_font, (220, 220, 220), pygame.Rect(right_rect.left + 12, right_rect.top + 12, right_rect.width - 24, right_rect.height - 24))

        controls = "Arrows: prev/next   Home/End: first/last   Q/Esc: quit"
        controls_surface = body_font.render(controls, True, (180, 180, 200))
        screen.blit(controls_surface, (20, args.window_height - 32))

        pygame.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main()
