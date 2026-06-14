# Data Dictionary — UCI Cleveland Heart Disease

Source: [UCI Machine Learning Repository — Heart Disease](https://archive.ics.uci.edu/ml/datasets/heart+disease)

| Variable | Type | Description | Clinical notes |
|----------|------|-------------|----------------|
| `age` | Numeric | Age in years | Cardiovascular risk increases with age |
| `sex` | Binary (1=male) | Biological sex | Males historically higher CHD prevalence in this cohort |
| `cp` | Ordinal (1–4) | Chest pain type | 1: typical angina, 2: atypical, 3: non-anginal, 4: asymptomatic |
| `trestbps` | Numeric | Resting blood pressure (mm Hg) | Hypertension is a modifiable CV risk factor |
| `chol` | Numeric | Serum cholesterol (mg/dL) | Lipid management / statin counseling relevance |
| `fbs` | Binary | Fasting blood sugar > 120 mg/dL | Diabetes screening proxy |
| `restecg` | Ordinal (0–2) | Resting electrocardiographic results | 0: normal, 1: ST-T abnormality, 2: LV hypertrophy |
| `thalach` | Numeric | Maximum heart rate achieved (exercise) | Exercise capacity indicator |
| `exang` | Binary | Exercise-induced angina (1=yes) | Ischemia signal during stress |
| `oldpeak` | Numeric | ST depression induced by exercise vs. rest | Ischemic ECG change |
| `slope` | Ordinal (1–3) | Slope of peak exercise ST segment | 1: upsloping, 2: flat, 3: downsloping |
| `ca` | Numeric (0–3) | Number of major vessels colored by fluoroscopy | Angiographic severity proxy; missing values present |
| `thal` | Ordinal (3,6,7) | Thalassemia defect type | 3: normal, 6: fixed defect, 7: reversible defect |
| `target` | Binary | Disease presence | UCI labels 1–4 recoded to 1; 0 = no disease |

## Derived columns

| Column | Definition |
|--------|------------|
| `target_label` | Human-readable outcome: "No disease" / "Disease" |

## Missing data

UCI encodes missing values as `?`. The pipeline imputes numeric features with the **training-set median** inside a scikit-learn `Pipeline` to prevent leakage.

## Cohort limitations

- 303 patients from Cleveland Clinic (1980s)
- Predominantly male
- Not representative of modern, diverse populations
- For portfolio and education only — not for clinical deployment
