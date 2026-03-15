"""Temporal smoothing, rep counting, and set tracking."""

from __future__ import annotations

import collections
from dataclasses import dataclass
import numpy as np


class VoteSmoother:
    def __init__(self, allowed_labels: set[str], window: int, min_votes: int):
        self.allowed_labels = allowed_labels
        self.buf = collections.deque(maxlen=window)
        self.min_votes = min_votes

    def reset(self) -> None:
        self.buf.clear()

    def update(self, label: str) -> str:
        if label not in self.allowed_labels:
            return "unknown"
        self.buf.append(label)
        counts = collections.Counter(self.buf)
        top_label, top_votes = counts.most_common(1)[0]
        return top_label if top_votes >= self.min_votes else "unknown"


class SignalRepCounter:
    def __init__(self, min_range: float = 12.0, cooldown_frames: int = 10):
        self.min_range = min_range
        self.cooldown_frames = cooldown_frames
        self.values = collections.deque(maxlen=120)
        self.phase = "unknown"
        self.reps = 0
        self.frame_idx = 0
        self.last_rep_frame = -10**9

    def reset(self) -> None:
        self.values.clear()
        self.phase = "unknown"
        self.reps = 0
        self.frame_idx = 0
        self.last_rep_frame = -10**9

    def update(self, value: float, invert: bool) -> int:
        self.frame_idx += 1
        signed_value = -value if invert else value
        self.values.append(float(signed_value))
        if len(self.values) < 20:
            return self.reps

        arr = np.array(self.values, dtype=float)
        low = np.percentile(arr, 20)
        high = np.percentile(arr, 80)
        value_range = high - low
        if value_range < self.min_range:
            return self.reps

        low_thr = low + 0.25 * value_range
        high_thr = low + 0.75 * value_range
        new_phase = self.phase
        if signed_value <= low_thr:
            new_phase = "low"
        elif signed_value >= high_thr:
            new_phase = "high"

        if self.phase == "low" and new_phase == "high":
            if (self.frame_idx - self.last_rep_frame) >= self.cooldown_frames:
                self.reps += 1
                self.last_rep_frame = self.frame_idx
        self.phase = new_phase
        return self.reps


class ConsecutiveExerciseSmoother:
    def __init__(self, required_streak: int = 10):
        self.required_streak = required_streak
        self.active_label = "unknown"
        self.candidate_label = "unknown"
        self.candidate_count = 0

    def reset(self) -> None:
        self.active_label = "unknown"
        self.candidate_label = "unknown"
        self.candidate_count = 0

    def update(self, label: str) -> str:
        if label == self.active_label:
            self.candidate_label = label
            self.candidate_count = 0
            return self.active_label

        if label == "unknown":
            return self.active_label

        if label == self.candidate_label:
            self.candidate_count += 1
        else:
            self.candidate_label = label
            self.candidate_count = 1

        if self.candidate_count >= self.required_streak:
            self.active_label = label
            self.candidate_count = 0
        return self.active_label


class StageRepCounter:
    def __init__(self, cooldown_frames: int = 6):
        self.total_reps = 0
        self.phase = "idle"
        self.cooldown_frames = cooldown_frames
        self.frame_idx = 0
        self.last_rep_frame = -10**9

    def reset(self) -> None:
        self.total_reps = 0
        self.phase = "idle"
        self.frame_idx = 0
        self.last_rep_frame = -10**9

    def update(self, stage: str) -> bool:
        self.frame_idx += 1
        if "_" in stage:
            _, stage = stage.rsplit("_", 1)
        if stage == "up":
            if self.phase == "seen_down":
                if (self.frame_idx - self.last_rep_frame) >= self.cooldown_frames:
                    self.total_reps += 1
                    self.last_rep_frame = self.frame_idx
                self.phase = "seen_up"
                return True
            self.phase = "seen_up"
            return False
        if stage == "down":
            if self.phase == "seen_up":
                self.phase = "seen_down"
            return False
        if stage == "rest":
            self.phase = "idle"
        return False


@dataclass
class SetStatus:
    current_set: int = 1
    reps_in_set: int = 0
    total_reps: int = 0
    completed_sets: int = 0
    workout_done: bool = False


class SetTracker:
    def __init__(self, reps_per_set: int = 10, target_sets: int = 4):
        self.reps_per_set = reps_per_set
        self.target_sets = target_sets
        self.status = SetStatus()

    def reset(self) -> None:
        self.status = SetStatus()

    def update(self, total_reps: int) -> SetStatus:
        if total_reps <= self.status.total_reps or self.status.workout_done:
            return self.status

        delta = total_reps - self.status.total_reps
        self.status.total_reps = total_reps
        for _ in range(delta):
            self.status.reps_in_set += 1
            if self.status.reps_in_set >= self.reps_per_set:
                self.status.completed_sets += 1
                if self.status.completed_sets >= self.target_sets:
                    self.status.workout_done = True
                    self.status.current_set = self.target_sets
                    self.status.reps_in_set = self.reps_per_set
                else:
                    self.status.current_set = self.status.completed_sets + 1
                    self.status.reps_in_set = 0
        if self.status.workout_done:
            self.status.current_set = self.target_sets
            self.status.reps_in_set = self.reps_per_set
        return self.status
