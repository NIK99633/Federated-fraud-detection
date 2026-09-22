from typing import Dict

import tensorflow as tf
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context

from app.config import BATCH_SIZE, LEARNING_RATE, LOCAL_EPOCHS
from app.dataset import load_client_partition, read_metadata
from app.metrics import compute_classification_metrics
from app.model import build_model, set_global_seed, set_model_weights


class BankClient(NumPyClient):
    """One simulated bank participating in federated learning."""

    def __init__(self, bank_id: int, input_dim: int) -> None:
        self.bank_id = bank_id
        self.input_dim = input_dim
        self.x_train, self.y_train, self.x_val, self.y_val = load_client_partition(bank_id)
        self.model = build_model(input_dim=self.input_dim, learning_rate=LEARNING_RATE)

    def _class_weight(self) -> Dict[int, float]:
        labels = self.y_train.astype(int)
        negatives = max(int((labels == 0).sum()), 1)
        positives = max(int((labels == 1).sum()), 1)
        minority_weight = negatives / positives
        return {0: 1.0, 1: float(minority_weight)}

    def _set_learning_rate(self, learning_rate: float) -> None:
        try:
            self.model.optimizer.learning_rate.assign(learning_rate)
        except Exception:
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate)

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        try:
            set_model_weights(self.model, parameters)
            local_epochs = int(config.get("local_epochs", LOCAL_EPOCHS))
            batch_size = int(config.get("batch_size", BATCH_SIZE))
            learning_rate = float(config.get("learning_rate", LEARNING_RATE))
            self._set_learning_rate(learning_rate)
            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=1, restore_best_weights=True)
            ]
            history = self.model.fit(
                self.x_train,
                self.y_train,
                validation_data=(self.x_val, self.y_val),
                epochs=local_epochs,
                batch_size=batch_size,
                verbose=0,
                class_weight=self._class_weight(),
                callbacks=callbacks,
            )
            train_loss = float(history.history["loss"][-1])
            return self.model.get_weights(), len(self.x_train), {"train_loss": train_loss}
        except Exception as exc:
            raise RuntimeError(f"Bank client {self.bank_id} training failed: {exc}") from exc

    def evaluate(self, parameters, config):
        try:
            set_model_weights(self.model, parameters)
            loss = float(self.model.evaluate(self.x_val, self.y_val, verbose=0))
            y_prob = self.model.predict(self.x_val, verbose=0).ravel()
            metrics, _, _, _ = compute_classification_metrics(self.y_val, y_prob)
            return loss, len(self.x_val), metrics
        except Exception as exc:
            raise RuntimeError(f"Bank client {self.bank_id} evaluation failed: {exc}") from exc


def client_fn(context: Context):
    try:
        bank_id = int(context.node_config["partition-id"])
    except KeyError as exc:
        raise KeyError(
            "Flower node configuration does not include 'partition-id'. Run this project through the simulation launcher."
        ) from exc

    metadata = read_metadata()
    input_dim = int(metadata["input_dim"])
    set_global_seed()
    return BankClient(bank_id=bank_id, input_dim=input_dim).to_client()


app = ClientApp(client_fn=client_fn)
