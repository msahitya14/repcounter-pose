"""Shared configuration and label contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExerciseConfig:
    name: str
    display_name: str
    stage_up: str
    stage_down: str
    stage_rest: str
    signal_mode: str
    invert_signal: bool


EXERCISE_CONFIGS = {
    "squat": ExerciseConfig(
        name="squat",
        display_name="Squats",
        stage_up="squat_up",
        stage_down="squat_down",
        stage_rest="squat_rest",
        signal_mode="knee_angle",
        invert_signal=False,
    ),
    "pushup": ExerciseConfig(
        name="pushup",
        display_name="Pushups",
        stage_up="pushup_up",
        stage_down="pushup_down",
        stage_rest="pushup_rest",
        signal_mode="elbow_angle",
        invert_signal=False,
    ),
    "pullup": ExerciseConfig(
        name="pullup",
        display_name="Pullups",
        stage_up="pullup_up",
        stage_down="pullup_down",
        stage_rest="pullup_rest",
        signal_mode="elbow_angle",
        invert_signal=True,
    ),
    "situp": ExerciseConfig(
        name="situp",
        display_name="Situps",
        stage_up="situp_up",
        stage_down="situp_down",
        stage_rest="situp_rest",
        signal_mode="torso_angle",
        invert_signal=False,
    ),
    "jumpingjack": ExerciseConfig(
        name="jumpingjack",
        display_name="Jumping Jacks",
        stage_up="jumpingjack_up",
        stage_down="jumpingjack_down",
        stage_rest="jumpingjack_rest",
        signal_mode="arm_spread",
        invert_signal=False,
    ),
}

EXERCISE_LABELS = list(EXERCISE_CONFIGS.keys())
STAGE_LABELS = [
    cfg.stage_up for cfg in EXERCISE_CONFIGS.values()
] + [
    cfg.stage_down for cfg in EXERCISE_CONFIGS.values()
] + [
    cfg.stage_rest for cfg in EXERCISE_CONFIGS.values()
]
