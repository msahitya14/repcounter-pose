"""Live webcam pipeline integration."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2

from repcounter.config import EXERCISE_CONFIGS, EXERCISE_LABELS, STAGE_LABELS
from repcounter.data import landmarks_array_from_results
from repcounter.features import build_feature_dict, get_signal_value
from repcounter.models import TrainedClassifier, load_classifier, predict_label
from repcounter.tracking import SetStatus, SetTracker, SignalRepCounter, StageRepCounter, VoteSmoother
from repcounter.ui import draw_overlay, draw_pose
from stage_labels import normalize_label, split_label


@dataclass
class LiveConfig:
    source: str
    exercise_model_path: str
    stage_model_path: str
    reps_per_set: int = 10
    target_sets: int = 4
    min_exercise_confidence: float = 0.55
    min_stage_confidence: float = 0.55
    exercise_window: int = 9
    exercise_min_votes: int = 4
    stage_window: int = 5
    stage_min_votes: int = 3
    rep_cooldown_frames: int = 6
    rep_feedback_seconds: float = 0.5
    lock_exercise: str = ""
    show_debug: bool = False


@dataclass
class ExerciseSessionState:
    stage_smoother: VoteSmoother
    stage_counter: StageRepCounter
    signal_counter: SignalRepCounter
    set_tracker: SetTracker
    active_stage: str = "unknown"
    rep_flash_until: float = 0.0
    rep_banner_until: float = 0.0


class LiveExerciseTracker:
    def __init__(self, config: LiveConfig):
        import mediapipe as mp

        self.config = config
        self.mp_pose = mp.solutions.pose
        self.exercise_model: Optional[TrainedClassifier] = None
        self.stage_model: Optional[TrainedClassifier] = None
        if config.exercise_model_path and Path(config.exercise_model_path).exists():
            self.exercise_model = load_classifier(config.exercise_model_path)
        if config.stage_model_path and Path(config.stage_model_path).exists():
            self.stage_model = load_classifier(config.stage_model_path)

        self.exercise_smoother = VoteSmoother(
            set(EXERCISE_LABELS),
            window=config.exercise_window,
            min_votes=config.exercise_min_votes,
        )
        self.stage_label_smoother = VoteSmoother(
            set(STAGE_LABELS),
            window=config.stage_window,
            min_votes=config.stage_min_votes,
        )
        self.exercise_states: Dict[str, ExerciseSessionState] = {
            name: ExerciseSessionState(
                stage_smoother=VoteSmoother(set(STAGE_LABELS), window=config.stage_window, min_votes=config.stage_min_votes),
                stage_counter=StageRepCounter(cooldown_frames=config.rep_cooldown_frames),
                signal_counter=SignalRepCounter(cooldown_frames=config.rep_cooldown_frames),
                set_tracker=SetTracker(reps_per_set=config.reps_per_set, target_sets=config.target_sets),
            )
            for name in EXERCISE_CONFIGS
        }
        self.active_exercise = "unknown"
        self.active_stage = "unknown"
        self.last_confidence = 0.0
        self.last_debug: dict[str, str] = {
            "raw_exercise": "unknown",
            "raw_stage": "unknown",
            "smooth_exercise": "unknown",
            "smooth_stage": "unknown",
            "stage_reps": "0",
            "signal_reps": "0",
        }

    def reset_for_exercise_change(self) -> None:
        self.stage_label_smoother.reset()
        self.active_stage = "unknown"
        state = self.exercise_states.get(self.active_exercise)
        if state is not None:
            state.stage_smoother.reset()
            state.active_stage = "unknown"

    def _session_totals(self) -> dict[str, int]:
        return {
            exercise: self.exercise_states[exercise].set_tracker.status.total_reps
            for exercise in EXERCISE_CONFIGS
        }

    def process_landmarks(self, landmarks_xyz, visibility, now_ts: float) -> tuple[str, str, SetStatus, float, bool]:
        feature_row = build_feature_dict(landmarks_xyz, visibility)

        predicted_exercise = "unknown"
        exercise_conf = 0.0
        if self.exercise_model is not None:
            label, exercise_conf, _ = predict_label(self.exercise_model, feature_row)
            if exercise_conf >= self.config.min_exercise_confidence:
                predicted_exercise = label
        if self.config.lock_exercise in EXERCISE_CONFIGS:
            predicted_exercise = self.config.lock_exercise
            exercise_conf = 1.0

        predicted_stage_label = "unknown"
        stage_conf = 0.0
        if self.stage_model is not None:
            stage_label, stage_conf, _ = predict_label(self.stage_model, feature_row)
            stage_label = normalize_label(stage_label)
            pred_ex, pred_stage = split_label(stage_label)
            if self.exercise_model is None and stage_conf >= self.config.min_stage_confidence:
                predicted_exercise = pred_ex
            if stage_conf >= self.config.min_stage_confidence and pred_ex != "unknown" and pred_stage in ("up", "down", "rest"):
                predicted_stage_label = stage_label

        stable_exercise = self.exercise_smoother.update(predicted_exercise)
        if stable_exercise in EXERCISE_CONFIGS and stable_exercise != self.active_exercise:
            self.active_exercise = stable_exercise
            self.reset_for_exercise_change()

        stable_stage_label = "unknown"
        state = self.exercise_states.get(self.active_exercise)
        if state is None:
            self.last_confidence = max(exercise_conf, stage_conf)
            return "unknown", "unknown", SetStatus(), self.last_confidence, False

        if self.stage_model is not None:
            pred_ex, pred_stage = split_label(predicted_stage_label)
            if self.exercise_model is None and stage_conf >= self.config.min_stage_confidence:
                predicted_exercise = pred_ex
                stable_exercise = self.exercise_smoother.update(predicted_exercise)
                if stable_exercise in EXERCISE_CONFIGS and stable_exercise != self.active_exercise:
                    self.active_exercise = stable_exercise
                    self.reset_for_exercise_change()
                    state = self.exercise_states[self.active_exercise]

            predicted_stage_for_active = "unknown"
            if stage_conf >= self.config.min_stage_confidence and pred_ex == self.active_exercise and pred_stage in ("up", "down", "rest"):
                predicted_stage_for_active = predicted_stage_label

            stable_stage_label = self.stage_label_smoother.update(predicted_stage_for_active)
            stable_stage_label = state.stage_smoother.update(stable_stage_label) if stable_stage_label != "unknown" else state.stage_smoother.update("unknown")
            if stable_stage_label != "unknown":
                _, self.active_stage = split_label(stable_stage_label)
            elif predicted_stage_for_active == "unknown" and self.active_exercise == "unknown":
                self.active_stage = "unknown"
        else:
            self.active_stage = "unknown"

        state.active_stage = self.active_stage

        stage_counter_input = stable_stage_label if stable_stage_label != "unknown" else "unknown"
        stage_reps_before = state.stage_counter.total_reps
        rep_completed = state.stage_counter.update(stage_counter_input)
        stage_reps = state.stage_counter.total_reps
        signal_reps = state.signal_counter.reps
        if self.active_exercise in EXERCISE_CONFIGS:
            signal_value = get_signal_value(self.active_exercise, landmarks_xyz)
            if signal_value is not None:
                signal_reps = state.signal_counter.update(signal_value, EXERCISE_CONFIGS[self.active_exercise].invert_signal)

        total_reps = max(stage_reps, signal_reps)
        if total_reps > stage_reps_before and not rep_completed:
            rep_completed = True
        if rep_completed:
            state.rep_flash_until = now_ts + self.config.rep_feedback_seconds
            state.rep_banner_until = now_ts + self.config.rep_feedback_seconds

        status = state.set_tracker.update(total_reps)
        self.last_confidence = max(exercise_conf, stage_conf)
        self.last_debug = {
            "raw_exercise": predicted_exercise,
            "raw_stage": predicted_stage_label,
            "smooth_exercise": self.active_exercise,
            "smooth_stage": stable_stage_label if stable_stage_label != "unknown" else state.active_stage,
            "stage_reps": str(stage_reps),
            "signal_reps": str(signal_reps),
        }
        return self.active_exercise, state.active_stage, status, self.last_confidence, rep_completed

    def run(self) -> None:
        cap = cv2.VideoCapture(int(self.config.source) if str(self.config.source).isdigit() else self.config.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source {self.config.source}")

        last_frame_ts = time.time()
        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                now_ts = time.time()
                fps = 1.0 / max(now_ts - last_frame_ts, 1e-6)
                last_frame_ts = now_ts
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                landmarks_xyz, visibility = landmarks_array_from_results(results)
                frame = draw_pose(frame, results)

                if landmarks_xyz is not None:
                    exercise, stage, status, confidence, _ = self.process_landmarks(landmarks_xyz, visibility, now_ts)
                else:
                    state = self.exercise_states.get(self.active_exercise)
                    if state is not None:
                        status = state.set_tracker.status
                        stage = state.active_stage
                    else:
                        status = SetStatus()
                        stage = "unknown"
                    exercise = self.active_exercise
                    confidence = self.last_confidence

                active_state = self.exercise_states.get(exercise)
                rep_flash = bool(active_state and now_ts <= active_state.rep_flash_until)
                rep_banner = bool(active_state and now_ts <= active_state.rep_banner_until)

                frame = draw_overlay(
                    frame,
                    exercise=exercise,
                    stage=stage,
                    set_number=status.current_set,
                    reps_in_set=status.reps_in_set,
                    total_reps=status.total_reps,
                    confidence=confidence,
                    fps=fps,
                    target_sets=self.config.target_sets,
                    target_reps=self.config.reps_per_set,
                    session_totals=self._session_totals(),
                    rep_flash=rep_flash,
                    rep_complete_banner=rep_banner,
                    debug_lines=[
                        f"raw ex: {self.last_debug['raw_exercise']}",
                        f"raw st: {self.last_debug['raw_stage']}",
                        f"sm ex: {self.last_debug['smooth_exercise']}",
                        f"sm st: {self.last_debug['smooth_stage']}",
                        f"stage reps: {self.last_debug['stage_reps']}",
                        f"signal reps: {self.last_debug['signal_reps']}",
                    ] if self.config.show_debug else None,
                )
                cv2.imshow("RepCounter", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Webcam index or path to video")
    parser.add_argument("--exercise_model", default="models/exercise_classifier.pkl")
    parser.add_argument("--stage_model", default="models/classifier.pkl")
    parser.add_argument("--clf", default=None, help="Backward-compatible alias for --stage_model")
    parser.add_argument("--reps_per_set", type=int, default=10)
    parser.add_argument("--target_sets", type=int, default=4)
    parser.add_argument("--min_exercise_confidence", type=float, default=0.55)
    parser.add_argument("--min_stage_confidence", type=float, default=0.55)
    parser.add_argument("--exercise_window", type=int, default=9)
    parser.add_argument("--exercise_min_votes", type=int, default=4)
    parser.add_argument("--stage_window", type=int, default=5)
    parser.add_argument("--stage_min_votes", type=int, default=3)
    parser.add_argument("--rep_cooldown_frames", type=int, default=6)
    parser.add_argument("--rep_feedback_seconds", type=float, default=0.5)
    parser.add_argument("--lock_exercise", default="", choices=["", "squat", "pushup", "pullup", "situp", "jumpingjack"])
    parser.add_argument("--show_debug", action="store_true")
    return parser
