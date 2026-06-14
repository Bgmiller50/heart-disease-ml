"""Download and load the UCI Cleveland heart disease dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import requests

from src.config import (
    COLUMN_NAMES,
    DATA_PROCESSED,
    DATA_RAW,
    LEGACY_PARQUET_FILENAME,
    PROCESSED_FILENAME,
    RAW_FILENAME,
    UCI_URL,
)


def download_raw(dest_dir: Path | None = None) -> Path:
    dest_dir = dest_dir or DATA_RAW
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / RAW_FILENAME

    if out_path.exists():
        print(f"Raw file already exists: {out_path}")
        return out_path

    print(f"Downloading from {UCI_URL} ...")
    response = requests.get(UCI_URL, timeout=60)
    response.raise_for_status()
    out_path.write_bytes(response.content)
    print(f"Saved to {out_path}")
    return out_path


def load_raw(path: Path | None = None) -> pd.DataFrame:
    path = path or (DATA_RAW / RAW_FILENAME)
    if not path.exists():
        download_raw(path.parent)

    df = pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values="?")
    # UCI: 0 = no disease; 1–4 = presence (severity). Binary for screening-style ML.
    df["target"] = (df["target"] > 0).astype(int)
    df["target_label"] = df["target"].map({0: "No disease", 1: "Disease"})
    return df


def save_processed(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or (DATA_PROCESSED / PROCESSED_FILENAME)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Processed data saved: {path}")
    return path


def _read_processed_file(path: Path) -> pd.DataFrame:
    """Read CSV (preferred). Parquet only for legacy files; rebuild on Arrow errors."""
    if path.suffix == ".csv":
        return pd.read_csv(path)

    try:
        return pd.read_parquet(path)
    except Exception as exc:
        print(f"Could not read {path} ({exc!r}). Rebuilding from raw...")
        path.unlink(missing_ok=True)
        raise


def load_processed(path: Path | None = None) -> pd.DataFrame:
    path = path or (DATA_PROCESSED / PROCESSED_FILENAME)
    legacy_parquet = DATA_PROCESSED / LEGACY_PARQUET_FILENAME

    if path.exists():
        return _read_processed_file(path)

    # One-time migration: parquet from an older run → CSV
    if legacy_parquet.exists():
        try:
            df = _read_processed_file(legacy_parquet)
            save_processed(df, path)
            return df
        except Exception:
            legacy_parquet.unlink(missing_ok=True)

    df = load_raw()
    save_processed(df, path)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Download UCI heart disease data")
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    raw_path = download_raw()
    if args.download_only:
        return

    df = load_raw(raw_path)
    save_processed(df)
    print(f"Rows: {len(df)}, columns: {list(df.columns)}")
    print(df["target_label"].value_counts())


if __name__ == "__main__":
    main()
