"""Dataset loading helpers for stage and exercise models."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from repcounter.features import build_feature_dict, MEDIAPIPE_LANDMARK_NAMES
from stage_labels import normalize_label, split_label


def _normalize_stage_series(series: pd.Series) -> pd.Series:
    normalized = series.astype(str).map(normalize_label)
    invalid = []
    for label in normalized.unique():
        ex, stage = split_label(label)
        if ex == "unknown" or stage == "unknown":
            invalid.append(label)
    if invalid:
        raise ValueError(f"Unrecognized stage labels: {sorted(invalid)[:10]}")
    return normalized


def _load_pose_feature_tables(data_dir: Path) -> pd.DataFrame:
    labels = pd.read_csv(data_dir / "labels.csv")
    labels.columns = [c.strip() for c in labels.columns]
    labels["pose"] = _normalize_stage_series(labels["pose"])

    merged = labels.reset_index(drop=True).copy()
    for name, file_name in {
        "landmarks": "landmarks.csv",
        "angles": "angles.csv",
        "dist3d": "3d_distances.csv",
        "distxyz": "xyz_distances.csv",
    }.items():
        path = data_dir / file_name
        if not path.exists():
            continue
        df = pd.read_csv(path).reset_index(drop=True)
        df.columns = [c.strip() for c in df.columns]
        if "pose_id" in df.columns and "pose_id" in merged.columns:
            rename = {c: f"{name}__{c}" for c in df.columns if c != "pose_id"}
            merged = merged.merge(df.rename(columns=rename), on="pose_id", how="left")
        else:
            rename = {c: f"{name}__{c}" for c in df.columns}
            merged = pd.concat([merged, df.rename(columns=rename)], axis=1)
    return merged


def _landmark_rows_to_feature_frame(landmarks_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    for _, row in landmarks_df.iterrows():
        landmarks = np.zeros((len(MEDIAPIPE_LANDMARK_NAMES), 3), dtype=np.float32)
        for idx, name in enumerate(MEDIAPIPE_LANDMARK_NAMES):
            landmarks[idx, 0] = float(row[f"x_{name}"])
            landmarks[idx, 1] = float(row[f"y_{name}"])
            landmarks[idx, 2] = float(row[f"z_{name}"])
        rows.append(build_feature_dict(landmarks))
    return pd.DataFrame(rows)


def load_stage_feature_csv_dir(data_dir: Union[str, Path]) -> Tuple[pd.DataFrame, pd.Series]:
    data_dir = Path(data_dir)
    merged = _load_pose_feature_tables(data_dir)
    landmarks = pd.read_csv(data_dir / "landmarks.csv")
    landmarks.columns = [c.strip() for c in landmarks.columns]
    features = _landmark_rows_to_feature_frame(landmarks)
    return features, merged["pose"].reset_index(drop=True)


def load_exercise_feature_csv_dir(data_dir: Union[str, Path]) -> Tuple[pd.DataFrame, pd.Series]:
    data_dir = Path(data_dir)
    merged = _load_pose_feature_tables(data_dir)
    landmarks = pd.read_csv(data_dir / "landmarks.csv")
    landmarks.columns = [c.strip() for c in landmarks.columns]
    features = _landmark_rows_to_feature_frame(landmarks)
    exercises = []
    for label in merged["pose"]:
        exercise, _ = split_label(label)
        if exercise == "unknown":
            raise ValueError(f"Could not derive exercise label from {label}")
        exercises.append(exercise)
    return features, pd.Series(exercises, name="exercise")


def load_flat_stage_dataset(csv_path: Union[str, Path]) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    label_col = "class" if "class" in df.columns else "pose"
    y = _normalize_stage_series(df[label_col])
    X = df.drop(columns=[label_col])
    return X, y


def load_flat_exercise_dataset(csv_path: Union[str, Path]) -> Tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    label_col = "class" if "class" in df.columns else "exercise"
    y = df[label_col].astype(str).str.strip().str.lower()
    X = df.drop(columns=[label_col])
    return X, y


def build_landmark_dataset_from_image_root(image_root: Union[str, Path]) -> Tuple[pd.DataFrame, pd.Series]:
    import cv2
    import mediapipe as mp

    image_root = Path(image_root)
    rows: list[dict[str, float]] = []
    labels: list[str] = []
    pose = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
    try:
        for class_dir in sorted(p for p in image_root.iterdir() if p.is_dir()):
            label = class_dir.name.strip().lower()
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    continue
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb)
                if result.pose_landmarks is None:
                    continue
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in result.pose_landmarks.landmark], dtype=np.float32)
                visibility = np.array([lm.visibility for lm in result.pose_landmarks.landmark], dtype=np.float32)
                rows.append(build_feature_dict(landmarks, visibility))
                labels.append(label)
    finally:
        pose.close()

    if not rows:
        raise ValueError(f"No pose landmarks could be extracted from {image_root}")
    return pd.DataFrame(rows), pd.Series(labels, name="exercise")


def landmarks_array_from_results(results) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if results.pose_landmarks is None:
        return None, None
    landmarks = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark], dtype=np.float32)
    visibility = np.array([lm.visibility for lm in results.pose_landmarks.landmark], dtype=np.float32)
    return landmarks, visibility
