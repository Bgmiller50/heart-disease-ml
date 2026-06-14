"""Exploratory data analysis — summary stats and figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import FEATURE_LABELS, FIGURES_DIR, rel_path
from src.load_data import load_processed
from src.train import FEATURE_COLS


def data_quality_summary(df: pd.DataFrame) -> dict:
    """Missing values, dtypes, class balance, and numeric ranges."""
    missing = df[FEATURE_COLS + ["target"]].isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    target_counts = df["target"].value_counts().to_dict()
    target_pct = (df["target"].value_counts(normalize=True) * 100).round(1).to_dict()

    numeric_summary = (
        df[FEATURE_COLS]
        .describe()
        .round(2)
        .to_dict()
    )

    return {
        "n_rows": int(len(df)),
        "n_features": len(FEATURE_COLS),
        "missing_counts": {k: int(v) for k, v in missing.items() if v > 0},
        "missing_pct": {k: float(v) for k, v in missing_pct.items() if v > 0},
        "target_counts": {str(k): int(v) for k, v in target_counts.items()},
        "target_pct": {str(k): float(v) for k, v in target_pct.items()},
        "numeric_summary": numeric_summary,
    }


def plot_class_balance(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["target_label"].value_counts()
    counts.plot(kind="bar", ax=ax, color=["#4C78A8", "#E45756"])
    ax.set_title("Class balance — coronary heart disease")
    ax.set_xlabel("Outcome")
    ax.set_ylabel("Patients (n)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 2, str(v), ha="center", fontsize=10)
    fig.tight_layout()
    path = out_dir / "class_balance.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_missing_values(df: pd.DataFrame, out_dir: Path) -> Path | None:
    missing = df[FEATURE_COLS].isna().sum()
    missing = missing[missing > 0]
    if missing.empty:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    missing.sort_values(ascending=True).plot(kind="barh", ax=ax, color="#E45756")
    ax.set_title("Missing values by feature")
    ax.set_xlabel("Count")
    fig.tight_layout()
    path = out_dir / "missing_values.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_clinical_features(df: pd.DataFrame, out_dir: Path) -> Path:
    """Lipids, BP, and fasting glucose proxy by disease status."""
    pharm_cols = ["chol", "trestbps", "fbs"]
    plot_df = df.melt(
        id_vars="target_label",
        value_vars=pharm_cols,
        var_name="feature",
        value_name="value",
    )
    plot_df["feature"] = plot_df["feature"].map(FEATURE_LABELS)

    g = sns.catplot(
        data=plot_df,
        x="target_label",
        y="value",
        col="feature",
        kind="box",
        sharey=False,
        height=3.5,
        aspect=1.1,
        hue="target_label",
        palette=["#4C78A8", "#E45756"],
        legend=False,
    )
    g.set_titles("{col_name}")
    g.fig.suptitle(
        "Pharmacy-relevant features by disease status",
        y=1.04,
        fontsize=12,
    )
    path = out_dir / "clinical_features_by_outcome.png"
    g.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(g.fig)
    return path


def plot_correlation_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    corr = df[FEATURE_COLS + ["target"]].corr()
    labels = [FEATURE_LABELS.get(c, c) for c in corr.columns]

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        corr,
        cmap="RdBu_r",
        center=0,
        ax=ax,
        square=True,
        xticklabels=labels,
        yticklabels=labels,
        annot=False,
    )
    ax.set_title("Feature correlation matrix (including target)")
    fig.tight_layout()
    path = out_dir / "correlation_heatmap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_age_chol_scatter(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=df,
        x="age",
        y="chol",
        hue="target_label",
        alpha=0.75,
        palette=["#4C78A8", "#E45756"],
        ax=ax,
    )
    ax.set_title("Age vs. cholesterol by disease status")
    ax.set_xlabel(FEATURE_LABELS["age"])
    ax.set_ylabel(FEATURE_LABELS["chol"])
    fig.tight_layout()
    path = out_dir / "age_chol_scatter.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def run_eda(df: pd.DataFrame | None = None, out_dir: Path | None = None) -> dict:
    """Generate EDA summary JSON and all exploratory figures."""
    if df is None:
        df = load_processed()
    out_dir = out_dir or FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = data_quality_summary(df)
    figures = {
        "class_balance": rel_path(plot_class_balance(df, out_dir)),
        "clinical_features": rel_path(plot_clinical_features(df, out_dir)),
        "correlation_heatmap": rel_path(plot_correlation_heatmap(df, out_dir)),
        "age_chol_scatter": rel_path(plot_age_chol_scatter(df, out_dir)),
    }
    missing_path = plot_missing_values(df, out_dir)
    if missing_path:
        figures["missing_values"] = rel_path(missing_path)

    report = {"summary": summary, "figures": figures}
    report_path = out_dir.parent / "eda_summary.json"
    report_path.write_text(json.dumps(report, indent=2))

    print(f"EDA complete — {len(df)} rows, {summary['n_features']} features")
    print(f"Class balance: {summary['target_pct']}")
    print(f"Report saved: {report_path}")
    return report


def main() -> None:
    run_eda()


if __name__ == "__main__":
    main()
