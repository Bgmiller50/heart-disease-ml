"""Clinical-style evaluation: ROC, confusion matrix, sensitivity/specificity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import FIGURES_DIR, MODELS_DIR, RANDOM_STATE, TEST_SIZE, rel_path
from src.load_data import load_processed
from src.train import FEATURE_COLS


def clinical_metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    return {
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def youden_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Threshold maximizing sensitivity + specificity - 1 (Youden's J)."""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_j = 0.5, -1.0
    for t in thresholds:
        m = clinical_metrics_at_threshold(y_true, y_prob, t)
        j = m["sensitivity"] + m["specificity"] - 1
        if j > best_j:
            best_j = j
            best_t = t
    return best_t


def evaluate(model_path: Path | None = None) -> dict:
    model_path = model_path or (MODELS_DIR / "best_model.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"No model at {model_path}. Run: python -m src.train")

    df = load_processed()
    X = df[FEATURE_COLS]
    y = df["target"].values

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = joblib.load(model_path)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = float(roc_auc_score(y_test, y_prob))

    threshold = youden_threshold(y_test, y_prob)
    metrics = clinical_metrics_at_threshold(y_test, y_prob, threshold)
    metrics["roc_auc"] = auc

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
    ax.set_title("ROC — Heart disease (test set)")
    fig.tight_layout()
    roc_path = FIGURES_DIR / "roc_curve.png"
    fig.savefig(roc_path, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, (y_prob >= threshold).astype(int), ax=ax, cmap="Blues"
    )
    ax.set_title(f"Confusion matrix (threshold={threshold:.2f})")
    fig.tight_layout()
    cm_path = FIGURES_DIR / "confusion_matrix.png"
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)

    prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=8, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve (test set)")
    ax.legend()
    fig.tight_layout()
    cal_path = FIGURES_DIR / "calibration_curve.png"
    fig.savefig(cal_path, dpi=150)
    plt.close(fig)

    print(classification_report(y_test, (y_prob >= threshold).astype(int), target_names=["No", "Yes"]))
    print("\nClinical metrics (Youden threshold):")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    report = {
        "metrics": metrics,
        "figures": {
            "roc": rel_path(roc_path),
            "confusion_matrix": rel_path(cm_path),
            "calibration": rel_path(cal_path),
        },
    }
    out = MODELS_DIR / "evaluation_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nReport saved: {out}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=None)
    args = parser.parse_args()
    evaluate(args.model)


if __name__ == "__main__":
    main()
