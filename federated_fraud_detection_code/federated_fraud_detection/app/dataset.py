import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

from app.config import (
    DATASET_PATH,
    GLOBAL_TEST_PATH,
    METADATA_PATH,
    NUM_CLIENTS,
    PARTITIONS_DIR,
    RANDOM_STATE,
    SCALER_PATH,
    TEST_SIZE,
    ensure_directories,
)


def _bank_file(bank_id: int) -> Path:
    return PARTITIONS_DIR / f"bank_{bank_id}.npz"


def ensure_dataset_exists() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at '{DATASET_PATH}'. "
            "Download 'creditcard.csv' from Kaggle and place it in the data/ folder."
        )


def load_dataset() -> pd.DataFrame:
    ensure_dataset_exists()
    try:
        df = pd.read_csv(DATASET_PATH)
    except Exception as exc:
        raise RuntimeError(f"Failed to read dataset: {exc}") from exc

    required_columns = {"Class", "Time", "Amount"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}. "
            "Expected the standard Kaggle creditcard.csv schema."
        )
    if df.isnull().sum().sum() > 0:
        raise ValueError("Dataset contains missing values. Clean the dataset before training.")
    return df


def _save_scaler(scaler: StandardScaler | None, scaler_columns: List[str], feature_columns: List[str]) -> None:
    payload = {
        "scaler": scaler,
        "scaler_columns": scaler_columns,
        "feature_columns": feature_columns,
    }
    with open(SCALER_PATH, "wb") as file:
        pickle.dump(payload, file)


def _preprocess_train_test(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    feature_columns = [col for col in train_df.columns if col != "Class"]
    scaler_columns = [col for col in ["Time", "Amount"] if col in feature_columns]

    x_train_df = train_df[feature_columns].copy()
    x_test_df = test_df[feature_columns].copy()

    scaler = None
    if scaler_columns:
        scaler = StandardScaler()
        x_train_df.loc[:, scaler_columns] = scaler.fit_transform(x_train_df[scaler_columns])
        x_test_df.loc[:, scaler_columns] = scaler.transform(x_test_df[scaler_columns])

    _save_scaler(scaler, scaler_columns, feature_columns)

    y_train = train_df["Class"].to_numpy(dtype=np.int32)
    y_test = test_df["Class"].to_numpy(dtype=np.int32)
    x_train = x_train_df.to_numpy(dtype=np.float32)
    x_test = x_test_df.to_numpy(dtype=np.float32)
    return x_train, y_train, x_test, y_test, feature_columns


def _split_local_train_val(x_client: np.ndarray, y_client: np.ndarray, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stratify_labels = y_client if np.unique(y_client).size > 1 else None
    x_train, x_val, y_train, y_val = train_test_split(
        x_client,
        y_client,
        test_size=0.20,
        random_state=seed,
        stratify=stratify_labels,
    )
    return x_train, x_val, y_train, y_val


def _clear_old_partitions() -> None:
    for file_path in PARTITIONS_DIR.glob("bank_*.npz"):
        file_path.unlink(missing_ok=True)
    GLOBAL_TEST_PATH.unlink(missing_ok=True)
    METADATA_PATH.unlink(missing_ok=True)


def prepare_federated_data(force_recreate: bool = False) -> Dict:
    ensure_directories()
    ensure_dataset_exists()

    expected_files = [_bank_file(bank_id) for bank_id in range(NUM_CLIENTS)]
    expected_files.extend([GLOBAL_TEST_PATH, METADATA_PATH])

    if not force_recreate and all(path.exists() for path in expected_files):
        return read_metadata()

    if NUM_CLIENTS != 4:
        raise ValueError("This submission is designed for exactly 4 simulated banks.")

    _clear_old_partitions()
    df = load_dataset()

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["Class"],
    )

    x_train_full, y_train_full, x_test, y_test, feature_columns = _preprocess_train_test(train_df, test_df)

    splitter = StratifiedKFold(n_splits=NUM_CLIENTS, shuffle=True, random_state=RANDOM_STATE)

    client_sizes = []
    client_positive_counts = []

    for bank_id, (_, client_indices) in enumerate(splitter.split(x_train_full, y_train_full)):
        x_client = x_train_full[client_indices]
        y_client = y_train_full[client_indices]
        x_train, x_val, y_train, y_val = _split_local_train_val(x_client=x_client, y_client=y_client, seed=RANDOM_STATE + bank_id)

        np.savez_compressed(
            _bank_file(bank_id),
            x_train=x_train.astype(np.float32),
            y_train=y_train.astype(np.int32),
            x_val=x_val.astype(np.float32),
            y_val=y_val.astype(np.int32),
        )

        client_sizes.append(int(len(y_client)))
        client_positive_counts.append(int(y_client.sum()))

    np.savez_compressed(
        GLOBAL_TEST_PATH,
        x_test=x_test.astype(np.float32),
        y_test=y_test.astype(np.int32),
    )

    metadata = {
        "num_clients": NUM_CLIENTS,
        "input_dim": int(x_train_full.shape[1]),
        "feature_columns": feature_columns,
        "train_samples_total": int(len(y_train_full)),
        "test_samples_total": int(len(y_test)),
        "test_positive_count": int(y_test.sum()),
        "client_sizes": client_sizes,
        "client_positive_counts": client_positive_counts,
        "random_state": RANDOM_STATE,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    return metadata


def read_metadata() -> Dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError("Metadata file not found. Run data preparation first.")
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def load_client_partition(bank_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    partition_path = _bank_file(bank_id)
    if not partition_path.exists():
        raise FileNotFoundError(
            f"Partition file '{partition_path}' not found. Run the simulation launcher first."
        )
    try:
        data = np.load(partition_path)
        return data["x_train"], data["y_train"], data["x_val"], data["y_val"]
    except Exception as exc:
        raise RuntimeError(f"Failed to load partition for bank {bank_id}: {exc}") from exc


def load_global_test_data() -> Tuple[np.ndarray, np.ndarray]:
    if not GLOBAL_TEST_PATH.exists():
        raise FileNotFoundError("Global test set not found. Run the simulation launcher first.")
    try:
        data = np.load(GLOBAL_TEST_PATH)
        return data["x_test"], data["y_test"]
    except Exception as exc:
        raise RuntimeError(f"Failed to load global test set: {exc}") from exc
