from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

SCALER_FILE = "scaler_hybrid.pkl"
COLUMNS_FILE = "feature_columns.json"
DATASET_FILE = "dataset_hybrid_bbri_final.csv"

SEQ_LENGTH = 5
TARGET_COL = "Close"
DATE_COL = "Date"

FITUR_BASELINE = ["Close", "High", "Low", "Open", "Volume"]
FITUR_HYBRID = FITUR_BASELINE + ["news_count", "sentiment_score"]

MODELS = {
    "baseline": {
        "label": "Baseline",
        "model_file": "model_baseline.keras",
        "x_test": "X_test_base.npy",
        "y_test": "Y_test_base.npy",
        "features": FITUR_BASELINE,
    },
    "hybrid": {
        "label": "Hybrid",
        "model_file": "model_hybrid.keras",
        "x_test": "X_test_hybrid.npy",
        "y_test": "Y_test_hybrid.npy",
        "features": FITUR_HYBRID,
    },
}

HYPERPARAMETER = {
    "Unit Bi-LSTM": "50",
    "Dropout": "0.2",
    "Learning rate": "0.001",
    "Batch size": "32",
}