"""
Shared stage-label contract and normalization helpers.

Canonical label format:
  <exercise>_<stage>
Examples:
  squat_up, squat_down, squat_rest
"""

from __future__ import annotations

import re

CANONICAL_EXERCISES = (
    "squat",
    "pushup",
    "pullup",
    "situp",
    "jumpingjack",
)

CANONICAL_STAGES = ("up", "down", "rest")

EXERCISE_ALIASES = {
    "squat": "squat",
    "squats": "squat",
    "pushup": "pushup",
    "pushups": "pushup",
    "push_up": "pushup",
    "push_ups": "pushup",
    "pullup": "pullup",
    "pullups": "pullup",
    "pull_up": "pullup",
    "pull_ups": "pullup",
    "situp": "situp",
    "situps": "situp",
    "sit_up": "situp",
    "sit_ups": "situp",
    "jumpingjack": "jumpingjack",
    "jumpingjacks": "jumpingjack",
    "jumping_jack": "jumpingjack",
    "jumping_jacks": "jumpingjack",
}

STAGE_ALIASES = {
    "up": "up",
    "top": "up",
    "down": "down",
    "bottom": "down",
    "rest": "rest",
    "idle": "rest",
    "neutral": "rest",
}


def _clean_token(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def normalize_label(label: str) -> str:
    """
    Convert variants like 'squats_down' or 'jumping_jacks_up'
    into canonical '<exercise>_<stage>' format when possible.
    """
    ex, stage = split_label(label)
    if ex == "unknown" or stage == "unknown":
        return _clean_token(str(label)) or "unknown"
    return f"{ex}_{stage}"


def split_label(label: str) -> tuple[str, str]:
    """
    Returns (exercise, stage), each canonical when recognized.
    Unknown parts are returned as 'unknown'.
    """
    raw = _clean_token(str(label))
    if not raw:
        return "unknown", "unknown"

    parts = raw.split("_")
    if len(parts) == 1:
        ex = EXERCISE_ALIASES.get(parts[0], "unknown")
        return ex, "unknown"

    stage = STAGE_ALIASES.get(parts[-1], "unknown")
    exercise_raw = "_".join(parts[:-1])
    ex = EXERCISE_ALIASES.get(exercise_raw, EXERCISE_ALIASES.get(exercise_raw.replace("_", ""), "unknown"))
    return ex, stage


def all_stage_labels(include_rest: bool = True) -> list[str]:
    stages = CANONICAL_STAGES if include_rest else ("up", "down")
    return [f"{ex}_{stage}" for ex in CANONICAL_EXERCISES for stage in stages]
