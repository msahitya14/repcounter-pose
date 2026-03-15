"""
Train the stage classifier from stage-labeled pose data.

Supported inputs:
  1. A directory with labels.csv + feature CSVs
  2. A flat CSV with class/pose + landmark/angle features
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from repcounter.data import load_flat_stage_dataset, load_stage_feature_csv_dir
from repcounter.models import save_classifier, train_classifier


def _save_confusion(confusion_matrix, classes: list[str], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(classes)), max(5, len(classes))))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    im = ax.imshow(confusion_matrix, cmap="Blues")
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", color="white")
    ax.set_yticks(range(len(classes)))
    ax.set_yticklabels(classes, color="white")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(
                j, i, str(confusion_matrix[i, j]),
                ha="center", va="center",
                color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2 else "#aaaacc",
                fontsize=9,
            )
    ax.set_title("Stage Classifier Confusion Matrix", color="white", fontsize=13)
    ax.set_xlabel("Predicted", color="#aaaacc")
    ax.set_ylabel("True", color="#aaaacc")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--csv", default=None, help="Optional flat stage dataset CSV")
    parser.add_argument("--model_out", default="models/stage_classifier.pkl")
    args = parser.parse_args()

    if args.csv:
        X, y = load_flat_stage_dataset(args.csv)
    else:
        X, y = load_stage_feature_csv_dir(args.data_dir)

    result = train_classifier(X, y)
    save_classifier(result["model"], args.model_out)
    print(f"CV accuracy: {result['cv_mean']:.4f} ± {result['cv_std']:.4f}")
    print(result["report"])

    confusion_path = Path(args.model_out).with_suffix(".confusion.png")
    _save_confusion(result["confusion_matrix"], result["model"].classes, confusion_path)
    print(f"Saved model to {args.model_out}")
    print(f"Saved confusion matrix to {confusion_path}")


if __name__ == "__main__":
    main()
