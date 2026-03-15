"""Live webcam pipeline integration."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

from repcounter.config import EXERCISE_CONFIGS, EXERCISE_LABELS, STAGE_LABELS
from repcounter.data import landmarks_array_from_results
from repcounter.features import build_feature_dict, get_signal_value
from repcounter.models import TrainedClassifier, load_classifier, predict_label
from repcounter.tracking import SetTracker, SignalRepCounter, StageRepCounter, VoteSmoother
from repcounter.ui import draw_overlay, draw_pose
from stage_labels import normalize_label, split_label


@dataclass
class LiveConfig:
    source: str
    exercise_model_path: str
    stage_model_path: str
    reps_per_set: int = 10
    target_sets: int = 3
    min_exercise_confidence: float = 0.55
    min_stage_confidence: float = 0.55
    rep_cooldown_frames: int = 10


class LiveExerciseTracker:
    def __init__(self, config: LiveConfig):
        import mediapipe as mp

        self.config = config
        self.mp_pose = mp.solutions.pose
        self.exercise_model: Optional[TrainedClassifier] = None
        self.stage_model: Optional[TrainedClassifier] = None
        if Path(config.exercise_model_path).exists():
            self.exercise_model = load_classifier(config.exercise_model_path)
        if Path(config.stage_model_path).exists():
            self.stage_model = load_classifier(config.stage_model_path)

        self.exercise_smoother = VoteSmoother(set(EXERCISE_LABELS), window=9, min_votes=4)
        self.stage_smoother = VoteSmoother(set(STAGE_LABELS), window=5, min_votes=3)
        self.stage_counter = StageRepCounter(cooldown_frames=config.rep_cooldown_frames)
        self.signal_counter = SignalRepCounter(cooldown_frames=config.rep_cooldown_frames)
        self.set_tracker = SetTracker(reps_per_set=config.reps_per_set, target_sets=config.target_sets)
        self.active_exercise = "unknown"
        self.active_stage = "unknown"
        self.last_confidence = 0.0

    def reset_for_exercise_change(self) -> None:
        self.stage_smoother.reset()
        self.stage_counter.reset()
        self.signal_counter.reset()
        self.set_tracker.reset()
        self.active_stage = "unknown"

    def process_landmarks(self, landmarks_xyz, visibility, now_ts: float) -> tuple[str, str, int, int, int, float]:
        feature_row = build_feature_dict(landmarks_xyz, visibility)

        predicted_exercise = "unknown"
        exercise_conf = 0.0
        if self.exercise_model is not None:
            label, exercise_conf, _ = predict_label(self.exercise_model, feature_row)
            if exercise_conf >= self.config.min_exercise_confidence:
                predicted_exercise = label
        stable_exercise = self.exercise_smoother.update(predicted_exercise)
        if stable_exercise in EXERCISE_CONFIGS and stable_exercise != self.active_exercise:
            self.active_exercise = stable_exercise
            self.reset_for_exercise_change()

        predicted_stage = "unknown"
        stage_conf = 0.0
        if self.stage_model is not None:
            stage_label, stage_conf, _ = predict_label(self.stage_model, feature_row)
            stage_label = normalize_label(stage_label)
            pred_ex, _ = split_label(stage_label)
            if self.exercise_model is None and stage_conf >= self.config.min_stage_confidence:
                predicted_exercise = pred_ex
                stable_exercise = self.exercise_smoother.update(predicted_exercise)
                if stable_exercise in EXERCISE_CONFIGS and stable_exercise != self.active_exercise:
                    self.active_exercise = stable_exercise
                    self.reset_for_exercise_change()
            if stage_conf >= self.config.min_stage_confidence and pred_ex == self.active_exercise:
                predicted_stage = stage_label

        stable_stage = self.stage_smoother.update(predicted_stage)
        if stable_stage != "unknown":
            self.active_stage = stable_stage
        elif predicted_stage == "unknown" and self.active_exercise == "unknown":
            self.active_stage = "unknown"

        stage_reps = self.stage_counter.update(stable_stage)
        signal_reps = 0
        if self.active_exercise in EXERCISE_CONFIGS:
            signal_value = get_signal_value(self.active_exercise, landmarks_xyz)
            if signal_value is not None:
                signal_reps = self.signal_counter.update(signal_value, EXERCISE_CONFIGS[self.active_exercise].invert_signal)

        total_reps = stage_reps if self.stage_model is not None else signal_reps
        status = self.set_tracker.update(total_reps, now_ts)
        self.last_confidence = max(exercise_conf, stage_conf)
        return (
            self.active_exercise,
            self.active_stage,
            status.current_set,
            status.reps_in_set,
            status.total_reps,
            self.last_confidence,
        )

    def run(self) -> None:
        cap = cv2.VideoCapture(int(self.config.source) if str(self.config.source).isdigit() else self.config.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open source {self.config.source}")

        with self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)
                landmarks_xyz, visibility = landmarks_array_from_results(results)
                frame = draw_pose(frame, results)

                if landmarks_xyz is not None:
                    exercise, stage, set_number, reps_in_set, total_reps, confidence = self.process_landmarks(
                        landmarks_xyz, visibility, time.time()
                    )
                else:
                    exercise, stage, set_number, reps_in_set, total_reps, confidence = (
                        self.active_exercise, self.active_stage, self.set_tracker.status.current_set,
                        self.set_tracker.status.reps_in_set, self.set_tracker.status.total_reps, self.last_confidence
                    )

                frame = draw_overlay(
                    frame,
                    exercise=exercise,
                    stage=stage,
                    set_number=set_number,
                    reps_in_set=reps_in_set,
                    total_reps=total_reps,
                    confidence=confidence,
                )
                cv2.imshow("RepCounter", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cap.release()
        cv2.destroyAllWindows()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Webcam index or path to video")
    parser.add_argument("--exercise_model", default="models/exercise_classifier.pkl")
    parser.add_argument("--stage_model", default="models/stage_classifier.pkl")
    parser.add_argument("--clf", default=None, help="Backward-compatible alias for --stage_model")
    parser.add_argument("--reps_per_set", type=int, default=10)
    parser.add_argument("--target_sets", type=int, default=3)
    parser.add_argument("--min_exercise_confidence", type=float, default=0.55)
    parser.add_argument("--min_stage_confidence", type=float, default=0.55)
    parser.add_argument("--rep_cooldown_frames", type=int, default=10)
    return parser
