"""Shared sklearn model training and inference helpers."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class TrainedClassifier:
    pipeline: Pipeline
    label_encoder: LabelEncoder
    feature_columns: list[str]
    classes: list[str]


def _prepare_matrix(X: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    feature_columns = list(X.columns)
    matrix = X.values.astype(np.float32)
    col_median = np.nanmedian(matrix, axis=0)
    inds = np.where(np.isnan(matrix))
    matrix[inds] = np.take(col_median, inds[1])
    return matrix, feature_columns


def train_classifier(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> dict:
    matrix, feature_columns = _prepare_matrix(X)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y.values)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        )),
    ])

    n_splits = max(2, min(5, int(min(np.bincount(y_encoded)))))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    scores = cross_val_score(pipeline, matrix, y_encoded, cv=cv, scoring="accuracy")

    pipeline.fit(matrix, y_encoded)
    preds = pipeline.predict(matrix)
    report = classification_report(y_encoded, preds, target_names=label_encoder.classes_, zero_division=0)
    cm = confusion_matrix(y_encoded, preds)

    return {
        "model": TrainedClassifier(
            pipeline=pipeline,
            label_encoder=label_encoder,
            feature_columns=feature_columns,
            classes=list(label_encoder.classes_),
        ),
        "cv_mean": float(scores.mean()),
        "cv_std": float(scores.std()),
        "report": report,
        "confusion_matrix": cm,
    }


def save_classifier(model: TrainedClassifier, output_path: Union[str, Path]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump({
            "pipeline": model.pipeline,
            "label_encoder": model.label_encoder,
            "feature_columns": model.feature_columns,
            "classes": model.classes,
        }, f)


def load_classifier(path: Union[str, Path]) -> TrainedClassifier:
    with Path(path).open("rb") as f:
        data = pickle.load(f)
    return TrainedClassifier(
        pipeline=data["pipeline"],
        label_encoder=data["label_encoder"],
        feature_columns=data.get("feature_columns", data.get("feat_cols", [])),
        classes=list(data.get("classes", data["label_encoder"].classes_)),
    )


def predict_label(model: TrainedClassifier, feature_row: dict[str, float]) -> tuple[str, float, np.ndarray]:
    x = np.array([feature_row.get(col, 0.0) for col in model.feature_columns], dtype=np.float32).reshape(1, -1)
    probas = model.pipeline.predict_proba(x)[0]
    idx = int(np.argmax(probas))
    return model.label_encoder.classes_[idx], float(probas[idx]), probas
