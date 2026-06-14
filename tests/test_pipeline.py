"""Smoke tests for reproducible pipeline stages."""

from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.config import COLUMN_NAMES, MODELS_DIR, PROJECT_ROOT
from src.eda import data_quality_summary, run_eda
from src.load_data import load_processed
from src.train import FEATURE_COLS, train_and_select


@pytest.fixture
def df() -> pd.DataFrame:
    return load_processed()


def test_processed_data_loads(df: pd.DataFrame) -> None:
    assert len(df) == 303
    assert "target" in df.columns
    assert set(FEATURE_COLS).issubset(df.columns)


def test_target_is_binary(df: pd.DataFrame) -> None:
    assert set(df["target"].unique()).issubset({0, 1})


def test_eda_summary(df: pd.DataFrame) -> None:
    summary = data_quality_summary(df)
    assert summary["n_rows"] == 303
    assert summary["n_features"] == 13
    assert "0" in summary["target_counts"]
    assert "1" in summary["target_counts"]


def test_train_and_select(df: pd.DataFrame) -> None:
    model, name, meta = train_and_select(df)
    assert name in ("logistic_regression", "random_forest")
    assert meta["best_roc_auc"] > 0.5
    assert "cv_roc_auc_mean" in meta
    assert model.predict_proba(df[FEATURE_COLS].head(5)).shape == (5, 2)


def test_eda_generates_figures(df: pd.DataFrame, tmp_path: Path) -> None:
    out = tmp_path / "figures"
    report = run_eda(df, out_dir=out)
    assert (out / "class_balance.png").exists()
    assert (out / "correlation_heatmap.png").exists()
    assert len(report["figures"]) >= 4


def test_model_artifacts_exist_if_trained() -> None:
    model_path = MODELS_DIR / "best_model.joblib"
    if not model_path.exists():
        pytest.skip("Model not trained — run: make train")
    model = joblib.load(model_path)
    assert hasattr(model, "predict_proba")


def test_project_structure() -> None:
    required = [
        PROJECT_ROOT / "src" / "load_data.py",
        PROJECT_ROOT / "src" / "train.py",
        PROJECT_ROOT / "src" / "evaluate.py",
        PROJECT_ROOT / "src" / "eda.py",
        PROJECT_ROOT / "src" / "interpret.py",
        PROJECT_ROOT / "src" / "run_pipeline.py",
        PROJECT_ROOT / "docs" / "DATA_DICTIONARY.md",
        PROJECT_ROOT / "notebooks" / "01_end_to_end_analysis.ipynb",
    ]
    for path in required:
        assert path.exists(), f"Missing: {path}"
