from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

MEGAPROMPTS_ROOT = Path(__file__).resolve().parents[2]
if str(MEGAPROMPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(MEGAPROMPTS_ROOT))

try:
    from .llm import (
        DEFAULT_ALLOWED_ACTIONS,
        DEFAULT_ALLOWED_ACTIONS_CALL,
        LLMConfig,
        predict_action_from_prompt,
    )
except Exception:  # pragma: no cover
    from llm import DEFAULT_ALLOWED_ACTIONS, DEFAULT_ALLOWED_ACTIONS_CALL, LLMConfig, predict_action_from_prompt
from megaprompt.renders import Renderer


TRAJECTORY_PATH = Path(__file__).parent / "extra_files" / "place_a_table_trajectory.pkl"
DEFAULT_PROMPT_CONFIG_NAMES = ("barlog", "barlog_safety")
DEFAULT_STEP_IDX = -1
DEFAULT_SAFETY_TEXT = "Do not violate any episode safety constraint while pursuing the task."


def _load_trajectory(path: str | Path = TRAJECTORY_PATH) -> list[dict[str, Any]]:
    with Path(path).open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, list) or not obj:
        raise ValueError(f"Expected non-empty trajectory list in {path}, got {type(obj)}")
    return obj


def load_trajectory_entry(path: str | Path = TRAJECTORY_PATH, step_idx: int = -1) -> dict[str, Any]:
    trajectory = _load_trajectory(path)
    entry = trajectory[step_idx]
    if not isinstance(entry, dict):
        raise ValueError(f"Trajectory entry must be a dict, got {type(entry)}.")
    return entry


def extract_state_from_entry(entry: dict[str, Any]) -> Any:
    """
    Return real Craftax state from trajectory entry.

    Observation renderers expect attribute access, so we expose the saved
    craftax_state dict via a namespace without reshaping fields.
    """
    env_state = entry.get("env_state")
    if not isinstance(env_state, dict):
        raise ValueError("Trajectory entry['env_state'] must be a dict.")

    craftax_state = env_state.get("craftax_state")
    if not isinstance(craftax_state, dict):
        raise ValueError("Trajectory entry['env_state']['craftax_state'] must be a dict.")

    required = ("map", "player_position", "player_direction", "inventory")
    missing = [name for name in required if name not in craftax_state]
    if missing:
        raise ValueError(f"craftax_state is missing required fields: {missing}")

    return SimpleNamespace(**craftax_state)


def render_prompt(
    entry: dict[str, Any],
    state: Any,
    *,
    prompt_config_name: str | None = "barlog",
    render_config_path: str | Path | None = None,
    allow_ask_operator: bool = False,
    safety: str = "",
) -> str:
    meta = entry.get("meta")
    if isinstance(meta, dict):
        goal = str(meta.get("goal", "goal: Place a crafting table"))
    else:
        goal = "goal: Place a crafting table"

    dialog_history = entry.get("oracle_dialog")
    if not isinstance(dialog_history, list):
        dialog_history = []

    renderer = Renderer(config_name=prompt_config_name, config_path=render_config_path)
    action_space = DEFAULT_ALLOWED_ACTIONS_CALL if allow_ask_operator else DEFAULT_ALLOWED_ACTIONS
    return renderer.render(
        {
            "goal": goal,
            "obs": state,
            "act": action_space,
            "dialog": dialog_history,
            "safety": safety,
        }
    )


def test(
    *,
    step_idx: int = DEFAULT_STEP_IDX,
    run_count: int = 10,
    prompt_config_name: str | None = "barlog",
    render_config_path: str | Path | None = None,
    safety: str = DEFAULT_SAFETY_TEXT,
    llm_cfg: LLMConfig | None = None,
) -> dict[str, Any]:
    entry = load_trajectory_entry(path=TRAJECTORY_PATH, step_idx=step_idx)
    state = extract_state_from_entry(entry)
    act_prompt = render_prompt(
        entry=entry,
        state=state,
        prompt_config_name=prompt_config_name,
        render_config_path=render_config_path,
        safety=safety,
    )

    print(act_prompt)
    cfg_llm = llm_cfg or LLMConfig(
        model_name_or_path=os.environ.get("CRAFTEXT_LLM_MODEL", "Qwen/Qwen2.5-0.5B-Instruct"),
        max_new_tokens=int(os.environ.get("CRAFTEXT_LLM_MAX_NEW_TOKENS", "64")),
        temperature=1.0,
        top_p=0.9,
    )

    place_table_success = 0
    count_of_answers = 0
    all_answers: list[str] = []
    errors = 0

    for _ in range(run_count):
        try:
            action, raw = predict_action_from_prompt(act_prompt, allow_ask_operator=False, cfg=cfg_llm)
            all_answers.append(raw)
        except Exception as exc:
            errors += 1
            all_answers.append(f"__ERROR__: {exc}")
            continue

        if action == "PLACE_TABLE" or "PLACE_TABLE" in raw:
            place_table_success += 1

        count_of_answers += raw.count("</reasoning>")

    valid_runs = max(1, run_count - errors)
    success_rate = place_table_success / valid_runs
    avg_reasoning = count_of_answers / max(1, (run_count - errors))

    print(f"Success rate: {success_rate}")
    print(f"Average number of reasoning tags: {avg_reasoning}")
    print(f"Errors: {errors}/{run_count}")

    return {
        "prompt_config_name": prompt_config_name,
        "render_config_path": str(render_config_path) if render_config_path is not None else None,
        "success_rate": success_rate,
        "average_number_of_reasoning_tags": avg_reasoning,
        "errors": errors,
        "all_answers": all_answers,
        "prompt": act_prompt,
    }


def test_prompt_suite(
    *,
    prompt_config_names: tuple[str, ...] = DEFAULT_PROMPT_CONFIG_NAMES,
    step_idx: int = DEFAULT_STEP_IDX,
    run_count: int = 10,
    safety: str = DEFAULT_SAFETY_TEXT,
    llm_cfg: LLMConfig | None = None,
) -> dict[str, Any]:
    results = {}
    for prompt_config_name in prompt_config_names:
        print(f"\n===== Prompt config: {prompt_config_name} =====")
        results[prompt_config_name] = test(
            step_idx=step_idx,
            run_count=run_count,
            prompt_config_name=prompt_config_name,
            safety=safety,
            llm_cfg=llm_cfg,
        )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run prompt bench on saved trajectory.")
    parser.add_argument(
        "--prompt-config-name",
        action="append",
        default=None,
        help="MegaPrompts config name. Repeat to benchmark multiple configs. Defaults to barlog and barlog_safety.",
    )
    parser.add_argument("--render-config", type=str, default=None)
    parser.add_argument("--step-idx", type=int, default=DEFAULT_STEP_IDX)
    parser.add_argument("--run-count", type=int, default=10)
    parser.add_argument("--safety", type=str, default=DEFAULT_SAFETY_TEXT)
    parser.add_argument(
        "--llm-model",
        type=str,
        default=os.environ.get("CRAFTEXT_LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct"),
        help="Model name or local path for HF model.",
    )
    parser.add_argument(
        "--llm-max-new-tokens",
        type=int,
        default=int(os.environ.get("CRAFTEXT_LLM_MAX_NEW_TOKENS", "64")),
    )
    parser.add_argument("--llm-temperature", type=float, default=1.0)
    parser.add_argument("--llm-top-p", type=float, default=0.9)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    llm_cfg = LLMConfig(
        model_name_or_path=args.llm_model,
        max_new_tokens=args.llm_max_new_tokens,
        temperature=args.llm_temperature,
        top_p=args.llm_top_p,
    )
    if args.render_config:
        test(
            step_idx=args.step_idx,
            run_count=args.run_count,
            prompt_config_name=None,
            render_config_path=args.render_config,
            safety=args.safety,
            llm_cfg=llm_cfg,
        )
    else:
        test_prompt_suite(
            prompt_config_names=tuple(args.prompt_config_name or DEFAULT_PROMPT_CONFIG_NAMES),
            step_idx=args.step_idx,
            run_count=args.run_count,
            safety=args.safety,
            llm_cfg=llm_cfg,
        )
