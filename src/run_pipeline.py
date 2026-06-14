"""Run the full end-to-end heart disease ML pipeline."""

from __future__ import annotations

import argparse
import time

from src.eda import run_eda
from src.evaluate import evaluate
from src.interpret import run_interpretation
from src.load_data import load_processed, save_processed
from src.train import save_artifacts, train_and_select


def run_all(retrain: bool = False) -> None:
    start = time.perf_counter()
    print("=" * 60)
    print("HEART DISEASE ML — END-TO-END PIPELINE")
    print("=" * 60)

    print("\n[1/5] Data ingestion & processing")
    df = load_processed()
    save_processed(df)

    print("\n[2/5] Exploratory data analysis")
    eda_report = run_eda(df)

    print("\n[3/5] Model training & selection")
    model, name, meta = train_and_select(df)
    save_artifacts(model, name, meta)

    print("\n[4/5] Clinical evaluation")
    eval_report = evaluate()

    print("\n[5/5] Model interpretation")
    interpret_report = run_interpretation()

    elapsed = time.perf_counter() - start
    print("\n" + "=" * 60)
    print(f"Pipeline complete in {elapsed:.1f}s")
    print(f"  Best model:     {meta['best_model']}")
    print(f"  Test ROC-AUC:   {meta['best_roc_auc']:.3f}")
    print(f"  CV ROC-AUC:     {meta.get('cv_roc_auc_mean', 0):.3f}")
    print(f"  EDA figures:    {len(eda_report['figures'])}")
    print(f"  Eval threshold: {eval_report['metrics']['threshold']:.2f}")
    print(f"  Top feature:    {interpret_report['top_features'][0]['label']}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full ML pipeline")
    parser.add_argument("--retrain", action="store_true", help="Force model retrain")
    args = parser.parse_args()
    run_all(retrain=args.retrain)


if __name__ == "__main__":
    main()
