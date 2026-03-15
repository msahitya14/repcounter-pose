"""Train an exercise-only classifier from stage-labeled pose data."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

from repcounter.features import build_feature_dict, MEDIAPIPE_LANDMARK_NAMES
from stage_labels import normalize_label, split_label


def load_exercise_dataset(data_dir: str):
    root = Path(data_dir)
    labels = pd.read_csv(root / "labels.csv")
    labels.columns = [c.strip() for c in labels.columns]
    labels["pose"] = labels["pose"].astype(str).map(normalize_label)

    landmarks = pd.read_csv(root / "landmarks.csv")
    landmarks.columns = [c.strip() for c in landmarks.columns]

    rows = []
    exercises = []
    for idx, row in landmarks.iterrows():
        lm = np.zeros((len(MEDIAPIPE_LANDMARK_NAMES), 3), dtype=np.float32)
        for j, name in enumerate(MEDIAPIPE_LANDMARK_NAMES):
            lm[j, 0] = float(row[f"x_{name}"])
            lm[j, 1] = float(row[f"y_{name}"])
            lm[j, 2] = float(row[f"z_{name}"])
        rows.append(build_feature_dict(lm))
        exercise, _ = split_label(labels.iloc[idx]["pose"])
        if exercise == "unknown":
            raise ValueError(f"Could not derive exercise label from {labels.iloc[idx]['pose']}")
        exercises.append(exercise)
    return pd.DataFrame(rows), pd.Series(exercises, name="exercise")


def train(data_dir: str, model_out: str):
    X, y = load_exercise_dataset(data_dir)
    feature_columns = list(X.columns)
    matrix = X.values.astype(np.float32)

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.values)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    n_splits = max(2, min(5, int(min(np.bincount(y_encoded)))))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, matrix, y_encoded, cv=cv, scoring="accuracy")
    print(f"CV accuracy: {scores.mean():.4f} +/- {scores.std():.4f}")

    pipeline.fit(matrix, y_encoded)
    preds = pipeline.predict(matrix)
    print(classification_report(y_encoded, preds, target_names=label_encoder.classes_, zero_division=0))

    out = Path(model_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump({
            "pipeline": pipeline,
            "label_encoder": label_encoder,
            "feature_columns": feature_columns,
            "classes": list(label_encoder.classes_),
        }, f)
    print(f"Saved exercise classifier -> {model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--model_out", default="models/exercise_classifier.pkl")
    args = parser.parse_args()
    train(args.data_dir, args.model_out)
