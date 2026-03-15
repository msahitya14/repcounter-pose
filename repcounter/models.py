"""Shared sklearn model loading and inference helpers."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


@dataclass
class TrainedClassifier:
    pipeline: Pipeline
    label_encoder: LabelEncoder
    feature_columns: list[str]
    classes: list[str]


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
