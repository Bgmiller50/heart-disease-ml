"""Train heart disease classifiers with a leakage-safe train/test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import COLUMN_NAMES, MODELS_DIR, RANDOM_STATE, TEST_SIZE
from src.load_data import load_processed


FEATURE_COLS = [c for c in COLUMN_NAMES if c != "target"]


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                FEATURE_COLS,
            )
        ]
    )


def make_models() -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=300,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def train_and_select(df: pd.DataFrame) -> tuple[Pipeline, str, dict]:
    X = df[FEATURE_COLS]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    results: dict[str, dict] = {}
    best_name = ""
    best_auc = -1.0
    best_model: Pipeline | None = None

    for name, pipeline in make_models().items():
        pipeline.fit(X_train, y_train)
        proba = pipeline.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, proba))
        results[name] = {"roc_auc": auc}
        print(f"{name}: ROC-AUC = {auc:.3f}")
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_model = pipeline

    assert best_model is not None
    meta = {
        "best_model": best_name,
        "best_roc_auc": best_auc,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "feature_columns": FEATURE_COLS,
    }
    return best_model, best_name, meta


def save_artifacts(
    model: Pipeline, model_name: str, meta: dict, out_dir: Path | None = None
) -> None:
    out_dir = out_dir or MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / "best_model.joblib")
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved model ({model_name}) to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrain", action="store_true", help="Force retrain")
    args = parser.parse_args()

    model_path = MODELS_DIR / "best_model.joblib"
    if model_path.exists() and not args.retrain:
        print(f"Model exists at {model_path}. Use --retrain to overwrite.")
        return

    df = load_processed()
    model, name, meta = train_and_select(df)
    save_artifacts(model, name, meta)


if __name__ == "__main__":
    main()
