# Build Project 1 From Scratch

Use this guide to **rebuild the entire heart disease project yourself**. The finished code in `src/` is your **answer key** — only peek when stuck.

**How to practice**

1. Work in `practice_build/` (empty starter — see below).
2. Follow phases in order (~2 hours/day).
3. Run the **Verify** command at the end of each phase.
4. Compare with `../src/` if something fails.

**Reset and try again**

```bash
cd practice_build
bash ../scripts/reset_practice_build.sh
```

---

## Phase 0: Empty project (Day 1, ~30 min)

### What you are building

A reproducible clinical ML repo:

```
practice_build/
├── requirements.txt
├── data/raw/
├── data/processed/
├── models/
├── reports/figures/
├── notebooks/
└── src/
    ├── config.py
    ├── load_data.py
    ├── train.py
    └── evaluate.py
```

### Steps

```bash
cd "/Users/barringtonmiller/Data Science Learning/projects/01_heart_disease/practice_build"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create empty folders:

```bash
mkdir -p data/raw data/processed models reports/figures notebooks src
touch src/__init__.py
```

### Verify

```bash
python -c "import pandas, sklearn, requests; print('OK')"
```

---

## Phase 1: `config.py` (Day 1, ~45 min)

**Purpose:** One file for paths, URLs, column names — no magic strings scattered in the codebase.

### What to type

Create `src/config.py`:

```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_FILENAME = "processed.cleveland.data"
PROCESSED_FILENAME = "heart_disease.csv"

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

RANDOM_STATE = 42
TEST_SIZE = 0.2

COLUMN_NAMES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]
```

**Why `parents[1]`?** `config.py` lives in `src/`, so one level up is the project root.

### Verify

```bash
python -c "from src.config import COLUMN_NAMES; print(len(COLUMN_NAMES))"
# Expect: 14
```

---

## Phase 2: `load_data.py` (Day 2, ~2 hr)

**Purpose:** Download UCI data → clean → save CSV → load anywhere.

### Functions to implement (in order)

| Function | Job |
|----------|-----|
| `download_raw()` | HTTP GET to `data/raw/`; skip if file exists |
| `load_raw()` | Read CSV, name columns, `?` → NaN, binary `target` |
| `save_processed()` | Write CSV |
| `load_processed()` | Read CSV or build from raw |
| `main()` | CLI entry when run as module |

### Key logic (you must understand, not just copy)

**1. Download**

```python
import requests
response = requests.get(UCI_URL, timeout=60)
response.raise_for_status()
out_path.write_bytes(response.content)
```

**2. Read CSV (no header in UCI file)**

```python
df = pd.read_csv(path, header=None, names=COLUMN_NAMES, na_values="?")
```

**3. Binary target (clinical endpoint)**

UCI: `0` = no disease, `1–4` = disease severity.

```python
df["target"] = (df["target"] > 0).astype(int)
df["target_label"] = df["target"].map({0: "No disease", 1: "Disease"})
```

**4. Processed CSV cache**

```python
df.to_csv(path, index=False)
return pd.read_csv(path)
```

### `main()` block

```python
if __name__ == "__main__":
    main()
```

### Verify

```bash
python -m src.load_data
```

Expect:

- `data/raw/processed.cleveland.data` exists
- `data/processed/heart_disease.csv` exists
- Printed row count **303**
- Class counts ~164 no disease / ~139 disease

```bash
python -c "from src.load_data import load_processed; df=load_processed(); print(df.shape, df.isna().sum().sum())"
```

---

## Phase 3: EDA notebook (Day 3, ~2 hr)

**Purpose:** Explore data *before* modeling; connect features to pharmacy practice.

Create `notebooks/01_eda_and_modeling.ipynb` with these sections:

1. **Setup** — add project root to `sys.path`, import `load_processed`, seaborn theme.
2. **Inspect** — `head()`, `shape`, `isna().sum()`, target balance.
3. **Plot** — bar chart of `target_label`.
4. **Clinical plots** — boxplots for `chol`, `trestbps`, `fbs` by outcome.
5. **Correlation** — heatmap of numeric features.
6. **Notes** — markdown cell: 5 bullet clinical observations (your words).

### Notebook path trick

```python
from pathlib import Path
import sys
PROJECT_ROOT = Path.cwd().resolve()
if PROJECT_ROOT.name == "notebooks":
    PROJECT_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
```

Run all cells from `practice_build/` root or with kernel cwd = project root.

### Verify

- Notebook runs without error.
- At least 3 figures saved under `reports/figures/`.
- You can explain **one** missing-value column and **one** feature correlated with `target`.

---

## Phase 4: `train.py` (Day 4, ~2 hr)

**Purpose:** Train ≥2 models, pick best by ROC-AUC, save pipeline.

### Concepts you must apply

| Concept | Implementation |
|---------|----------------|
| Features vs target | `FEATURE_COLS` = all columns except `target` |
| Leakage-safe split | `train_test_split(..., stratify=y, random_state=42)` |
| Missing data | `SimpleImputer(strategy="median")` **inside** pipeline |
| Scaling | `StandardScaler` after imputer |
| Class imbalance | `class_weight="balanced"` |
| Compare models | Logistic regression vs random forest |
| Persist | `joblib.dump(pipeline, models/best_model.joblib)` |

### Pipeline structure

```python
Pipeline([
    ("preprocess", ColumnTransformer([("num", impute_scale_pipeline, FEATURE_COLS)])),
    ("clf", RandomForestClassifier(...)),
])
```

### `train_and_select` logic

1. Split `X`, `y`.
2. For each model: `fit` → `predict_proba` on test → `roc_auc_score`.
3. Keep model with highest AUC.
4. Save `model_meta.json` with name, AUC, `n_train`, `n_test`.

### Verify

```bash
python -m src.train
```

Expect:

- Printed AUC for both models (typically ~0.93–0.96).
- `models/best_model.joblib` and `models/model_meta.json` exist.

---

## Phase 5: `evaluate.py` (Day 5, ~2 hr)

**Purpose:** Clinical-style metrics, not only accuracy.

### Functions to implement

**`clinical_metrics_at_threshold(y_true, y_prob, threshold)`**

- `y_pred = (y_prob >= threshold).astype(int)`
- Confusion matrix → TP, FP, TN, FN
- Sensitivity = TP / (TP + FN)
- Specificity = TN / (TN + FP)
- PPV = TP / (TP + FP)
- NPV = TN / (TN + FN)

**`youden_threshold(y_true, y_prob)`**

- Try thresholds 0.01 … 0.99
- Pick threshold maximizing sensitivity + specificity − 1

**`evaluate()`**

1. Same `train_test_split` as training (same `RANDOM_STATE`, `TEST_SIZE`, `stratify`).
2. Load model with `joblib.load`.
3. ROC curve → save PNG.
4. Confusion matrix at Youden threshold → save PNG.
5. Calibration curve → save PNG.
6. Print `classification_report`.
7. Write `models/evaluation_report.json`.

### Verify

```bash
python -m src.evaluate
```

Expect figures in `reports/figures/` and printed sensitivity/specificity.

---

## Phase 6: Interpretation (Day 6, ~2 hr)

Add a notebook cell (after training):

```python
import joblib
pipe = joblib.load(MODELS_DIR / "best_model.joblib")
clf = pipe.named_steps["clf"]
```

- If Random Forest: `feature_importances_` bar chart.
- If Logistic Regression: coefficients + odds ratios discussion.

Write **half a page** in the notebook:

- Top 3 drivers of prediction.
- Pharmacy angle (lipids, BP, glucose).
- Why this is **not** deployment-ready (small n, old data, no external validation).

### Verify

You can explain results to a non-technical clinician in 2 minutes.

---

## Phase 7: Portfolio README (Day 7, ~2 hr)

Create `practice_build/README.md` with:

1. Problem statement (1 paragraph).
2. Data source + license/educational disclaimer.
3. How to run (`pip install`, three `python -m` commands).
4. Results table (AUC, sensitivity, specificity).
5. Limitations (5 bullets).
6. Screenshot or link to best figure.

### Final end-to-end verify

From `practice_build/`:

```bash
rm -rf data/raw/* data/processed/* models/* reports/figures/*
python -m src.load_data
python -m src.train --retrain
python -m src.evaluate
jupyter nbconvert --execute notebooks/01_eda_and_modeling.ipynb
```

If all succeed, you built the project from scratch.

---

## Self-check: Can you do this without the repo?

Before moving to Project 2, you should be able to **explain aloud**:

1. Where does the data come from and what does `target` mean?
2. Why binarize 1–4 into a single “disease” class?
3. Why impute in the pipeline, not before the split?
4. What is leakage and how did you avoid it?
5. Difference between ROC-AUC and sensitivity at a fixed threshold?
6. What is Youden’s index used for?

---

## When stuck

| Symptom | Check |
|---------|--------|
| `ModuleNotFoundError: src` | Run commands from `practice_build/`, not `src/` |
| Different metrics vs friend | Same `RANDOM_STATE`, same split, same threshold logic |
| AUC = 1.0 | Leakage — target in features? |
| `FileNotFoundError` model | Run `train` before `evaluate` |

**Answer key:** `../src/` in the main project folder.

---

## After you finish

1. Push `practice_build/` to GitHub as **your** portfolio repo.
2. Optionally delete `practice_build/` and rebuild once more from memory (best retention test).
3. Tell your tutor: **“Ready for Project 2”** — diabetes readmission / chronic disease.
