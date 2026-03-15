"""Drawing helpers for overlaying pose and workout status."""

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


def _fit_text_scale(text: str, font: int, max_width: int, base_scale: float, thickness: int) -> float:
    scale = base_scale
    while scale > 0.35:
        text_width = cv2.getTextSize(text, font, scale, thickness)[0][0]
        if text_width <= max_width:
            return scale
        scale -= 0.05
    return 0.35


def _draw_panel(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    fill_color: tuple[int, int, int],
    border_color: tuple[int, int, int],
    alpha: float = 0.82,
) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), fill_color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2, cv2.LINE_AA)


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
    fps: float,
    target_sets: int,
    target_reps: int,
    session_totals: dict[str, int],
    rep_flash: bool = False,
    rep_complete_banner: bool = False,
    debug_lines: list[str] | None = None,
) -> np.ndarray:
    h, w = frame.shape[:2]
    exercise_text = EXERCISE_CONFIGS[exercise].display_name if exercise in EXERCISE_CONFIGS else "Detecting..."
    stage_text = "WAITING"
    if stage not in ("unknown", ""):
        stage_text = stage.upper()

    margin = 16
    top_panel_height = max(220, int(h * 0.34))
    left_panel_width = max(350, int(w * 0.58))
    right_panel_width = max(170, int(w * 0.2))
    right_panel_x1 = max(left_panel_width + margin * 2, w - right_panel_width - margin)
    bottom_panel_height = 88

    rep_color = (167, 255, 87) if rep_flash else (229, 237, 247)
    rep_border = (60, 220, 90) if rep_flash else (49, 199, 181)

    _draw_panel(frame, margin, margin, left_panel_width, margin + top_panel_height, (12, 16, 28), rep_border)
    _draw_panel(frame, right_panel_x1, margin, w - margin, margin + 112, (12, 16, 28), (90, 126, 255))
    _draw_panel(frame, margin, h - bottom_panel_height - margin, w - margin, h - margin, (12, 16, 28), (49, 199, 181))

    title_font = cv2.FONT_HERSHEY_DUPLEX
    body_font = cv2.FONT_HERSHEY_SIMPLEX
    strong_font = cv2.FONT_HERSHEY_DUPLEX

    title = "AI Exercise Tracker"
    title_scale = _fit_text_scale(title, title_font, left_panel_width - 48, 0.95, 2)
    cv2.putText(frame, title, (margin + 18, margin + 34), title_font, title_scale, (239, 247, 255), 2, cv2.LINE_AA)

    exercise_line = f"Exercise: {exercise_text}"
    exercise_scale = _fit_text_scale(exercise_line, strong_font, left_panel_width - 48, 0.95, 2)
    cv2.putText(frame, exercise_line, (margin + 18, margin + 76), strong_font, exercise_scale, (117, 216, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Set: {set_number} / {target_sets}", (margin + 18, margin + 114), strong_font, 0.82, (238, 241, 247), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Reps: {reps_in_set} / {target_reps}", (margin + 18, margin + 150), strong_font, 0.82, (238, 241, 247), 2, cv2.LINE_AA)

    rep_label_color = (115, 215, 108) if rep_flash else (171, 182, 205)
    cv2.putText(frame, "Current Rep", (margin + 18, margin + 188), body_font, 0.7, rep_label_color, 2, cv2.LINE_AA)
    rep_value = str(reps_in_set)
    rep_scale = _fit_text_scale(rep_value, strong_font, int(left_panel_width * 0.28), 2.4, 5)
    rep_y = margin + 282
    cv2.putText(frame, rep_value, (margin + 18, rep_y), strong_font, rep_scale, rep_color, 5, cv2.LINE_AA)

    rep_target_text = f"/ {target_reps}"
    rep_target_scale = _fit_text_scale(rep_target_text, strong_font, left_panel_width - 180, 1.15, 2)
    cv2.putText(frame, rep_target_text, (margin + 132, rep_y - 8), strong_font, rep_target_scale, (229, 237, 247), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Stage: {stage_text}", (margin + 18, margin + top_panel_height - 48), strong_font, 0.82, (255, 199, 126), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Total Reps: {total_reps}", (margin + 18, margin + top_panel_height - 16), strong_font, 0.82, (202, 224, 255), 2, cv2.LINE_AA)

    if rep_complete_banner:
        cv2.putText(frame, "REP COMPLETE", (margin + 220, margin + 206), strong_font, 0.84, (90, 255, 120), 2, cv2.LINE_AA)

    info_x = right_panel_x1 + 16
    cv2.putText(frame, f"Confidence: {confidence:.2f}", (info_x, margin + 42), body_font, 0.66, (235, 241, 249), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.0f}", (info_x, margin + 78), body_font, 0.66, (235, 241, 249), 2, cv2.LINE_AA)

    summary = " | ".join(
        f"{EXERCISE_CONFIGS[name].display_name}: {session_totals.get(name, 0)}"
        for name in EXERCISE_CONFIGS
    )
    cv2.putText(frame, "Total Workout Progress", (margin + 12, h - bottom_panel_height + 20), strong_font, 0.72, (166, 230, 222), 2, cv2.LINE_AA)
    summary_scale = _fit_text_scale(summary, body_font, w - 64, 0.72, 2)
    cv2.putText(frame, summary, (margin + 12, h - 28), body_font, summary_scale, (231, 237, 242), 2, cv2.LINE_AA)

    if debug_lines:
        debug_height = 26 + 22 * len(debug_lines)
        _draw_panel(frame, right_panel_x1, margin + 124, w - margin, min(h - bottom_panel_height - margin - 8, margin + 124 + debug_height), (12, 16, 28), (235, 171, 78), alpha=0.78)
        cv2.putText(frame, "Debug", (right_panel_x1 + 16, margin + 148), strong_font, 0.6, (255, 214, 145), 2, cv2.LINE_AA)
        for idx, line in enumerate(debug_lines):
            cv2.putText(frame, line, (right_panel_x1 + 16, margin + 172 + idx * 20), body_font, 0.5, (232, 238, 244), 1, cv2.LINE_AA)
    return frame
