"""Drawing helpers for overlaying pose and tracking status."""

from __future__ import annotations

import cv2
import numpy as np

from repcounter.config import EXERCISE_CONFIGS


POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (0, 11), (0, 12),
]


def draw_pose(frame: np.ndarray, results, conf_thr: float = 0.5) -> np.ndarray:
    if results.pose_landmarks is None:
        return frame
    h, w = frame.shape[:2]
    pts = np.array([[lm.x * w, lm.y * h] for lm in results.pose_landmarks.landmark])
    vis = np.array([lm.visibility for lm in results.pose_landmarks.landmark])

    for a, b in POSE_CONNECTIONS:
        if vis[a] > conf_thr and vis[b] > conf_thr:
            cv2.line(frame, tuple(pts[a].astype(int)), tuple(pts[b].astype(int)), (60, 220, 255), 3, cv2.LINE_AA)
    for idx, pt in enumerate(pts):
        if vis[idx] > conf_thr:
            cv2.circle(frame, tuple(pt.astype(int)), 5, (0, 255, 140), -1, cv2.LINE_AA)
    return frame


def draw_overlay(
    frame: np.ndarray,
    exercise: str,
    stage: str,
    set_number: int,
    reps_in_set: int,
    total_reps: int,
    confidence: float,
) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()
    panel_w = min(420, w - 12)
    cv2.rectangle(overlay, (12, 12), (12 + panel_w, 164), (14, 18, 30), -1)
    cv2.rectangle(overlay, (12, 12), (12 + panel_w, 164), (46, 205, 180), 2)
    cv2.addWeighted(overlay, 0.84, frame, 0.16, 0, frame)

    exercise_text = EXERCISE_CONFIGS[exercise].display_name if exercise in EXERCISE_CONFIGS else "Detecting..."
    if stage != "unknown" and "_" in stage:
        _, stage_suffix = stage.rsplit("_", 1)
        stage_text = stage_suffix.title()
    else:
        stage_text = "Waiting"

    title_font = cv2.FONT_HERSHEY_TRIPLEX
    body_font = cv2.FONT_HERSHEY_DUPLEX

    cv2.putText(frame, "AI EXERCISE TRACKER", (28, 38), title_font, 0.65, (235, 248, 242), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Exercise: {exercise_text}", (28, 74), body_font, 0.92, (244, 222, 120), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Set: {set_number}", (28, 110), body_font, 0.84, (240, 244, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Reps: {reps_in_set}", (28, 146), body_font, 0.84, (64, 240, 190), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Stage: {stage_text}", (220, 110), body_font, 0.82, (255, 194, 112), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Total: {total_reps}", (220, 146), body_font, 0.78, (188, 210, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"{confidence * 100:.0f}%", (w - 110, 38), title_font, 0.58, (190, 198, 210), 1, cv2.LINE_AA)
    return frame
