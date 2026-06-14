from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_FILENAME = "processed.cleveland.data"
PROCESSED_FILENAME = "heart_disease.csv"
LEGACY_PARQUET_FILENAME = "heart_disease.parquet"  # older runs; ignored after migration

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

RANDOM_STATE = 42
TEST_SIZE = 0.2

# UCI Cleveland heart disease — 14 attributes (target last)
COLUMN_NAMES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "target",
]

FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Sex (1=male)",
    "cp": "Chest pain type",
    "trestbps": "Resting BP (mmHg)",
    "chol": "Serum cholesterol (mg/dL)",
    "fbs": "Fasting BS > 120 mg/dL",
    "restecg": "Resting ECG",
    "thalach": "Max heart rate",
    "exang": "Exercise angina",
    "oldpeak": "ST depression (oldpeak)",
    "slope": "ST slope",
    "ca": "Major vessels (fluoroscopy)",
    "thal": "Thalassemia",
}
