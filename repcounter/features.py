"""Pose feature engineering utilities."""

from __future__ import annotations

from typing import Optional

import numpy as np

from repcounter.config import EXERCISE_CONFIGS


MEDIAPIPE_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear", "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_pinky_1", "right_pinky_1",
    "left_index_1", "right_index_1", "left_thumb_2", "right_thumb_2",
    "left_hip", "right_hip", "left_knee", "right_knee",
    "left_ankle", "right_ankle", "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def angle3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    cos = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _safe_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def normalize_landmarks(landmarks_xyz: np.ndarray) -> np.ndarray:
    lm = np.asarray(landmarks_xyz, dtype=np.float32).copy()
    if len(lm) < 29:
        return lm

    hip_center = (lm[23] + lm[24]) / 2.0
    shoulder_center = (lm[11] + lm[12]) / 2.0
    shoulder_width = np.linalg.norm(lm[11] - lm[12])
    torso_length = np.linalg.norm(shoulder_center - hip_center)
    scale = max(shoulder_width, torso_length, 1e-6)
    return (lm - hip_center) / scale


def compute_joint_angles(landmarks_xyz: np.ndarray) -> dict[str, float]:
    lm = landmarks_xyz
    return {
        "right_elbow_right_shoulder_right_hip": angle3(lm[14], lm[12], lm[24]),
        "left_elbow_left_shoulder_left_hip": angle3(lm[13], lm[11], lm[23]),
        "right_knee_mid_hip_left_knee": angle3(lm[26], (lm[23] + lm[24]) / 2.0, lm[25]),
        "right_hip_right_knee_right_ankle": angle3(lm[24], lm[26], lm[28]),
        "left_hip_left_knee_left_ankle": angle3(lm[23], lm[25], lm[27]),
        "right_wrist_right_elbow_right_shoulder": angle3(lm[16], lm[14], lm[12]),
        "left_wrist_left_elbow_left_shoulder": angle3(lm[15], lm[13], lm[11]),
    }


def compute_distance_features(landmarks_xyz: np.ndarray) -> dict[str, float]:
    lm = landmarks_xyz
    return {
        "left_shoulder_left_wrist": _safe_distance(lm[11], lm[15]),
        "right_shoulder_right_wrist": _safe_distance(lm[12], lm[16]),
        "left_hip_left_ankle": _safe_distance(lm[23], lm[27]),
        "right_hip_right_ankle": _safe_distance(lm[24], lm[28]),
        "left_hip_left_wrist": _safe_distance(lm[23], lm[15]),
        "right_hip_right_wrist": _safe_distance(lm[24], lm[16]),
        "left_shoulder_left_ankle": _safe_distance(lm[11], lm[27]),
        "right_shoulder_right_ankle": _safe_distance(lm[12], lm[28]),
        "left_hip_right_wrist": _safe_distance(lm[23], lm[16]),
        "right_hip_left_wrist": _safe_distance(lm[24], lm[15]),
        "left_elbow_right_elbow": _safe_distance(lm[13], lm[14]),
        "left_knee_right_knee": _safe_distance(lm[25], lm[26]),
        "left_wrist_right_wrist": _safe_distance(lm[15], lm[16]),
        "left_ankle_right_ankle": _safe_distance(lm[27], lm[28]),
        "left_hip_avg_left_wrist_left_ankle": _safe_distance(lm[23], (lm[15] + lm[27]) / 2.0),
        "right_hip_avg_right_wrist_right_ankle": _safe_distance(lm[24], (lm[16] + lm[28]) / 2.0),
    }


def compute_landmark_features(landmarks_xyz: np.ndarray, visibility: Optional[np.ndarray] = None) -> dict[str, float]:
    del visibility
    features: dict[str, float] = {}
    for idx, name in enumerate(MEDIAPIPE_LANDMARK_NAMES):
        features[f"landmarks__x_{name}"] = float(landmarks_xyz[idx, 0])
        features[f"landmarks__y_{name}"] = float(landmarks_xyz[idx, 1])
        features[f"landmarks__z_{name}"] = float(landmarks_xyz[idx, 2])
    return features


def compute_axis_distance_features(landmarks_xyz: np.ndarray) -> dict[str, float]:
    lm = landmarks_xyz
    pairs = {
        "left_shoulder_left_wrist": (11, 15),
        "right_shoulder_right_wrist": (12, 16),
        "left_hip_left_ankle": (23, 27),
        "right_hip_right_ankle": (24, 28),
        "left_hip_left_wrist": (23, 15),
        "right_hip_right_wrist": (24, 16),
        "left_shoulder_left_ankle": (11, 27),
        "right_shoulder_right_ankle": (12, 28),
        "left_hip_right_wrist": (23, 16),
        "right_hip_left_wrist": (24, 15),
        "left_elbow_right_elbow": (13, 14),
        "left_knee_right_knee": (25, 26),
        "left_wrist_right_wrist": (15, 16),
        "left_ankle_right_ankle": (27, 28),
    }
    features: dict[str, float] = {}
    for name, (i, j) in pairs.items():
        diff = np.abs(lm[i] - lm[j])
        features[f"distxyz__x_{name}"] = float(diff[0])
        features[f"distxyz__y_{name}"] = float(diff[1])
        features[f"distxyz__z_{name}"] = float(diff[2])

    left_mid = np.abs(lm[23] - ((lm[15] + lm[27]) / 2.0))
    right_mid = np.abs(lm[24] - ((lm[16] + lm[28]) / 2.0))
    features["distxyz__x_left_hip_avg_left_wrist_left_ankle"] = float(left_mid[0])
    features["distxyz__y_left_hip_avg_left_wrist_left_ankle"] = float(left_mid[1])
    features["distxyz__z_left_hip_avg_left_wrist_left_ankle"] = float(left_mid[2])
    features["distxyz__x_right_hip_avg_right_wrist_right_ankle"] = float(right_mid[0])
    features["distxyz__y_right_hip_avg_right_wrist_right_ankle"] = float(right_mid[1])
    features["distxyz__z_right_hip_avg_right_wrist_right_ankle"] = float(right_mid[2])
    return features


def build_feature_dict(landmarks_xyz: np.ndarray, visibility: Optional[np.ndarray] = None) -> dict[str, float]:
    normalized = normalize_landmarks(landmarks_xyz)
    features = compute_landmark_features(normalized, visibility)
    features.update({f"angles__{k}": v for k, v in compute_joint_angles(normalized).items()})
    features.update({f"dist3d__{k}": v for k, v in compute_distance_features(normalized).items()})
    features.update(compute_axis_distance_features(normalized))
    return features


def get_signal_value(exercise: str, landmarks_xyz: np.ndarray) -> Optional[float]:
    if exercise not in EXERCISE_CONFIGS or len(landmarks_xyz) < 29:
        return None

    cfg = EXERCISE_CONFIGS[exercise]
    normalized = normalize_landmarks(landmarks_xyz)
    angles = compute_joint_angles(normalized)
    if cfg.signal_mode == "wrist_y":
        return float((normalized[15, 1] + normalized[16, 1]) / 2.0)
    if cfg.signal_mode == "elbow_angle":
        return float((angles["left_wrist_left_elbow_left_shoulder"] + angles["right_wrist_right_elbow_right_shoulder"]) / 2.0)
    if cfg.signal_mode == "knee_angle":
        return float(angles["left_hip_left_knee_left_ankle"])
    if cfg.signal_mode == "torso_angle":
        return float((angles["left_elbow_left_shoulder_left_hip"] + angles["right_elbow_right_shoulder_right_hip"]) / 2.0)
    return None
