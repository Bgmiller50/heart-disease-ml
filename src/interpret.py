"""Model interpretation — feature importance, coefficients, threshold sensitivity."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from src.config import FEATURE_LABELS, FIGURES_DIR, MODELS_DIR, RANDOM_STATE, TEST_SIZE, rel_path
from src.evaluate import clinical_metrics_at_threshold
from src.load_data import load_processed
from src.train import FEATURE_COLS


def feature_importance_table(model) -> pd.DataFrame:
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
        kind = "importance"
    elif hasattr(clf, "coef_"):
        values = np.abs(clf.coef_[0])
        kind = "abs_coefficient"
    else:
        raise ValueError("Model has no feature_importances_ or coef_")

    table = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "label": [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS],
            kind: values,
        }
    ).sort_values(kind, ascending=False)
    return table


def plot_feature_importance(model, out_dir: Path) -> Path:
    table = feature_importance_table(model)
    col = [c for c in table.columns if c not in ("feature", "label")][0]

    fig, ax = plt.subplots(figsize=(8, 6))
    plot_data = table.sort_values(col, ascending=True)
    ax.barh(plot_data["label"], plot_data[col], color="#4C78A8")
    title = (
        "Random Forest feature importance"
        if col == "importance"
        else "Logistic regression |coefficients|"
    )
    ax.set_title(title)
    ax.set_xlabel(col.replace("_", " ").title())
    fig.tight_layout()
    path = out_dir / "feature_importance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def threshold_sensitivity(y_true: np.ndarray, y_prob: np.ndarray) -> pd.DataFrame:
    rows = []
    for t in np.arange(0.1, 0.95, 0.05):
        m = clinical_metrics_at_threshold(y_true, y_prob, round(t, 2))
        rows.append(m)
    return pd.DataFrame(rows)


def plot_threshold_sensitivity(sensitivity_df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(
        sensitivity_df["threshold"],
        sensitivity_df["sensitivity"],
        marker="o",
        label="Sensitivity (recall)",
    )
    ax.plot(
        sensitivity_df["threshold"],
        sensitivity_df["specificity"],
        marker="s",
        label="Specificity",
    )
    ax.plot(
        sensitivity_df["threshold"],
        sensitivity_df["ppv"],
        marker="^",
        label="PPV (precision)",
        linestyle="--",
    )
    ax.set_xlabel("Classification threshold")
    ax.set_ylabel("Rate")
    ax.set_title("Threshold sensitivity — clinical operating points")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = out_dir / "threshold_sensitivity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def top_clinical_insights(importance_df: pd.DataFrame, n: int = 5) -> list[str]:
    col = [c for c in importance_df.columns if c not in ("feature", "label")][0]
    top = importance_df.head(n)
    insights = []
    for _, row in top.iterrows():
        insights.append(f"{row['label']} ({row['feature']}) — rank {len(insights) + 1}")
    return insights


def run_interpretation(model_path: Path | None = None, out_dir: Path | None = None) -> dict:
    model_path = model_path or (MODELS_DIR / "best_model.joblib")
    out_dir = out_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

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

    importance_df = feature_importance_table(model)
    sensitivity_df = threshold_sensitivity(y_test, y_prob)

    figures = {
        "feature_importance": rel_path(plot_feature_importance(model, out_dir)),
        "threshold_sensitivity": rel_path(plot_threshold_sensitivity(sensitivity_df, out_dir)),
    }

    report = {
        "top_features": importance_df.head(8).to_dict(orient="records"),
        "threshold_sensitivity": sensitivity_df.round(3).to_dict(orient="records"),
        "clinical_insights": top_clinical_insights(importance_df),
        "figures": figures,
    }

    out_path = MODELS_DIR / "interpretation_report.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"Interpretation saved: {out_path}")
    return report


def main() -> None:
    run_interpretation()


if __name__ == "__main__":
    main()
