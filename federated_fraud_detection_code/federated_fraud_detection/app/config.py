import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PARTITIONS_DIR = DATA_DIR / "partitions"
OUTPUTS_DIR = BASE_DIR / "outputs"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

DATASET_PATH = Path(os.getenv("FFD_DATASET_PATH", str(DATA_DIR / "creditcard.csv")))
NUM_CLIENTS = int(os.getenv("FFD_NUM_CLIENTS", "4"))
NUM_ROUNDS = int(os.getenv("FFD_NUM_ROUNDS", "10"))
LOCAL_EPOCHS = int(os.getenv("FFD_LOCAL_EPOCHS", "2"))
BATCH_SIZE = int(os.getenv("FFD_BATCH_SIZE", "256"))
LEARNING_RATE = float(os.getenv("FFD_LEARNING_RATE", "0.001"))
TEST_SIZE = float(os.getenv("FFD_TEST_SIZE", "0.20"))
RANDOM_STATE = int(os.getenv("FFD_RANDOM_STATE", "42"))

MODEL_PATH = ARTIFACTS_DIR / "global_fraud_model.keras"
SCALER_PATH = ARTIFACTS_DIR / "feature_scaler.pkl"

GLOBAL_TEST_PATH = PARTITIONS_DIR / "global_test.npz"
METADATA_PATH = PARTITIONS_DIR / "metadata.json"

ROUND_METRICS_JSON = OUTPUTS_DIR / "round_metrics.json"
ROUND_METRICS_CSV = OUTPUTS_DIR / "round_metrics.csv"
FINAL_METRICS_JSON = OUTPUTS_DIR / "final_metrics.json"
CONFUSION_MATRIX_PATH = OUTPUTS_DIR / "confusion_matrix.png"
ROC_CURVE_PATH = OUTPUTS_DIR / "roc_curve.png"


def ensure_directories() -> None:
    """Create all required folders if they do not exist."""
    for path in [DATA_DIR, PARTITIONS_DIR, OUTPUTS_DIR, ARTIFACTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
