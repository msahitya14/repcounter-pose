"""
train_classifier.py
====================
Trains an exercise pose classifier on your MediaPipe feature CSVs.

Expected files (all share the same pose_id index):
  data/landmarks.csv      – pose_id + 33 landmarks × x,y,z  (99 cols)
  data/angles.csv         – pose_id + joint angles
  data/3d_distances.csv   – pose_id + 3-D distances between landmark pairs
  data/xyz_distances.csv  – pose_id + per-axis distances
  data/labels.csv         – pose_id + pose  (e.g. "squat_down")

Outputs:
  models/classifier.pkl   – trained pipeline (scaler + GBM)
  models/confusion.png    – confusion matrix

Usage:
  python train_classifier.py --data_dir data/ --model_out models/classifier.pkl
"""

import argparse, os, pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

from stage_labels import normalize_label, split_label


# ── MediaPipe landmark names (33) ─────────────────────────────────────────────
MP_LANDMARKS = [
    "nose","left_eye_inner","left_eye","left_eye_outer",
    "right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear","mouth_left","mouth_right",
    "left_shoulder","right_shoulder","left_elbow","right_elbow",
    "left_wrist","right_wrist","left_pinky_1","right_pinky_1",
    "left_index_1","right_index_1","left_thumb_2","right_thumb_2",
    "left_hip","right_hip","left_knee","right_knee",
    "left_ankle","right_ankle","left_heel","right_heel",
    "left_foot_index","right_foot_index",
]


def load_and_merge(data_dir: str) -> pd.DataFrame:
    """Load all feature CSVs and merge on pose_id."""
    d = Path(data_dir)
    files = {
        "landmarks":    d / "landmarks.csv",
        "angles":       d / "angles.csv",
        "dist3d":       d / "3d_distances.csv",
        "distxyz":      d / "xyz_distances.csv",
    }
    labels_path = d / "labels.csv"

    labels = pd.read_csv(labels_path)
    labels.columns = [c.strip() for c in labels.columns]
    labels["pose"] = labels["pose"].astype(str).map(normalize_label)

    invalid = []
    for label in labels["pose"].unique():
        ex, stage = split_label(label)
        if ex == "unknown" or stage == "unknown":
            invalid.append(label)
    if invalid:
        bad = ", ".join(sorted(invalid)[:10])
        raise ValueError(
            "Unrecognized stage labels found after normalization. "
            f"Examples: {bad}"
        )

    merged = labels.copy()
    for name, path in files.items():
        if path.exists():
            df = pd.read_csv(path)
            df.columns = [c.strip() for c in df.columns]
            # Rename non-pose_id columns to avoid clashes
            rename = {c: f"{name}__{c}" for c in df.columns if c != "pose_id"}
            df = df.rename(columns=rename)
            merged = merged.merge(df, on="pose_id", how="left")
            print(f"  ✓ {name}: {len(df.columns)-1} features")
        else:
            print(f"  ⚠ {path} not found, skipping")

    return merged


def build_feature_matrix(merged: pd.DataFrame):
    """Drop id/label columns, return X array and label series."""
    drop_cols = {"pose_id", "pose"}
    feat_cols = [c for c in merged.columns if c not in drop_cols]
    X = merged[feat_cols].values.astype(np.float32)
    # Replace NaN with column median
    col_median = np.nanmedian(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_median, inds[1])
    return X, merged["pose"].values, feat_cols


def train(data_dir: str, model_out: str):
    print("📂 Loading data …")
    merged = load_and_merge(data_dir)
    print(f"   Total rows: {len(merged)}")

    X, y_raw, feat_cols = build_feature_matrix(merged)
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    classes = le.classes_
    print(f"   Features: {X.shape[1]}  |  Classes: {list(classes)}")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=300, max_depth=4,
            learning_rate=0.08, subsample=0.8,
            random_state=42
        )),
    ])

    print("🔁 Cross-validating (5-fold) …")
    n_splits = min(5, min(np.bincount(y)))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
    print(f"   CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    print("🏋️  Fitting final model …")
    pipe.fit(X, y)
    preds = pipe.predict(X)
    print("\n" + classification_report(y, preds, target_names=classes))

    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    with open(model_out, "wb") as f:
        pickle.dump({
            "pipeline":      pipe,
            "label_encoder": le,
            "classes":       classes,
            "feat_cols":     feat_cols,
        }, f)
    print(f"✅ Model saved → {model_out}")

    # Confusion matrix
    cm = confusion_matrix(y, preds)
    fig, ax = plt.subplots(figsize=(max(6, len(classes)), max(5, len(classes))))
    fig.patch.set_facecolor("#0f0f1a"); ax.set_facecolor("#0f0f1a")
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right", color="white")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes, color="white")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="white" if cm[i,j] > cm.max()/2 else "#aaaacc", fontsize=9)
    ax.set_title("Confusion Matrix", color="white", fontsize=13)
    ax.set_xlabel("Predicted", color="#aaaacc"); ax.set_ylabel("True", color="#aaaacc")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    out_png = Path(model_out).with_suffix(".confusion.png")
    plt.savefig(out_png, dpi=120, facecolor=fig.get_facecolor())
    plt.close()
    print(f"   Confusion matrix → {out_png}")
    return pipe, le, classes


# ── Inference API (used by rep_counter.py) ────────────────────────────────────
def load_model(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_pose(model_dict: dict, row: dict) -> tuple[str, float, np.ndarray]:
    """
    row: dict with keys matching feat_cols
    Returns (predicted_pose_label, confidence, all_probas)
    """
    pipe   = model_dict["pipeline"]
    le     = model_dict["label_encoder"]
    fcols  = model_dict["feat_cols"]
    x = np.array([row.get(c, 0.0) for c in fcols], dtype=np.float32).reshape(1, -1)
    proba  = pipe.predict_proba(x)[0]
    idx    = np.argmax(proba)
    return le.classes_[idx], float(proba[idx]), proba


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",  default="data/")
    p.add_argument("--model_out", default="models/classifier.pkl")
    args = p.parse_args()
    train(args.data_dir, args.model_out)
