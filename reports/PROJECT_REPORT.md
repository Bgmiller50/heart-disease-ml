# Project Report: Heart Disease Risk Prediction

**Author:** Barrington Miller  
**Date:** June 2025  
**Repository:** [github.com/Bgmiller50/heart-disease-ml](https://github.com/Bgmiller50/heart-disease-ml)

---

## 1. Executive Summary

This project builds an end-to-end machine learning workflow to predict coronary heart disease presence from 13 clinical features using the UCI Cleveland dataset (n=303). The pipeline ingests raw data, performs exploratory analysis, trains and compares two classifiers, evaluates performance with clinical metrics, and interprets model drivers.

**Key result:** Random Forest achieved **ROC-AUC 0.958** on the held-out test set with **89.3% sensitivity** and **97.0% specificity** at the Youden-optimal threshold.

> ⚠️ Not validated for clinical decision-making.

---

## 2. Business & Clinical Context

### Problem
Cardiovascular disease remains a leading cause of mortality. Risk stratification from routinely collected clinical variables supports earlier referral and counseling — particularly around lipids, blood pressure, and exercise tolerance.

### Stakeholders
- **Clinical analysts** — screening-style metrics (sensitivity, specificity, calibration)
- **Pharmacy / care management** — modifiable risk factors (chol, BP, glucose proxy)
- **Data science reviewers** — reproducible pipeline, leakage-safe preprocessing, interpretation

### Success criteria
| Criterion | Target | Outcome |
|-----------|--------|---------|
| Discrimination (ROC-AUC) | > 0.85 | ✅ 0.958 |
| Sensitivity at tuned threshold | > 80% | ✅ 89.3% |
| Reproducible pipeline | Single command | ✅ `make pipeline` |
| Interpretability | Top features ranked | ✅ Feature importance + threshold table |

---

## 3. Data

| Item | Detail |
|------|--------|
| Source | UCI Cleveland Heart Disease |
| Patients | 303 |
| Features | 13 clinical + 1 binary target |
| Missing values | Encoded as `?`; median imputation in pipeline |
| Class balance | ~54% disease / 46% no disease |

See [docs/DATA_DICTIONARY.md](../docs/DATA_DICTIONARY.md) for full variable definitions.

---

## 4. Methodology

### Pipeline architecture

```
Raw UCI data → clean & encode → EDA figures → train/test split (80/20, stratified)
    → Logistic Regression vs Random Forest (5-fold CV) → select best by test ROC-AUC
    → clinical evaluation (ROC, calibration, Youden threshold) → interpretation
```

### Modeling decisions
- **Preprocessing:** `SimpleImputer(median)` + `StandardScaler` inside `ColumnTransformer` — fit on train only
- **Class imbalance:** `class_weight="balanced"` for both models
- **Model selection:** Compare test ROC-AUC; 5-fold CV for stability check
- **Threshold:** Youden's J (maximize sensitivity + specificity − 1) — appropriate for screening framing

### What we deliberately did not do
- Deep learning (dataset too small)
- External validation cohort (not available)
- Causal inference (observational data only)

---

## 5. Results

### Model comparison

| Model | Test ROC-AUC | 5-Fold CV ROC-AUC |
|-------|--------------|-------------------|
| Logistic Regression | ~0.93 | ~0.89 |
| **Random Forest** | **0.958** | **~0.91** |

### Clinical metrics (test set, Youden threshold ≈ 0.64)

| Metric | Value |
|--------|-------|
| Sensitivity | 89.3% |
| Specificity | 97.0% |
| PPV | 96.2% |
| NPV | 91.4% |

### Top predictive features (Random Forest)
1. Maximum heart rate (`thalach`) — exercise capacity
2. ST depression (`oldpeak`) — ischemic ECG signal
3. Chest pain type (`cp`) — angina presentation
4. Number of major vessels (`ca`) — angiographic severity
5. Thalassemia defect (`thal`)

---

## 6. Recommendations

### For a care management analytics team
- Prioritize patients with **low exercise capacity**, **elevated ST depression**, and **atypical chest pain patterns** for follow-up — consistent with top model drivers.
- Use **probability outputs** with calibration checks before setting referral thresholds; default Youden threshold is a starting point, not a clinical standard.

### For model improvement (v2)
- Stratified k-fold reporting with confidence intervals
- SHAP values for individual prediction explanations
- Larger, modern, diverse cohort for external validation
- Separate models by sex given known epidemiological differences

---

## 7. Limitations

- Single-center 1980s cohort — poor generalizability
- n=303 — metrics have high variance
- Some ordinal features treated as numeric
- Missing `ca` values imputed — may bias vessel-related signal
- No prospective validation or fairness analysis across demographics

---

## 8. Reproducibility

```bash
make setup      # once: create venv + install deps
make pipeline   # full end-to-end run
make test       # smoke tests
```

Artifacts generated:
- `reports/eda_summary.json` + EDA figures
- `models/best_model.joblib` + `model_meta.json`
- `models/evaluation_report.json`
- `models/interpretation_report.json`
- `reports/figures/*.png`

---

## 9. Author

**Barrington Miller** — Healthcare & Life Sciences Data Scientist  
M.S. Data Science · B.S. Pharmacy · Austin, TX  
[GitHub](https://github.com/Bgmiller50) · barringtonmiller55@gmail.com
