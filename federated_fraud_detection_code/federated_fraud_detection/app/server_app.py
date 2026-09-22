from typing import Dict, List, Tuple

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from app.config import (
    BATCH_SIZE,
    CONFUSION_MATRIX_PATH,
    FINAL_METRICS_JSON,
    LEARNING_RATE,
    LOCAL_EPOCHS,
    MODEL_PATH,
    NUM_CLIENTS,
    NUM_ROUNDS,
    ROC_CURVE_PATH,
    ROUND_METRICS_CSV,
    ROUND_METRICS_JSON,
    ensure_directories,
)
from app.dataset import load_global_test_data, read_metadata
from app.metrics import compute_classification_metrics, save_confusion_matrix, save_roc_curve, save_round_history
from app.model import build_model, set_global_seed, set_model_weights

ROUND_HISTORY: List[Dict[str, float]] = []
BEST_AUC: float = -1.0


def reset_server_state() -> None:
    global ROUND_HISTORY, BEST_AUC
    ROUND_HISTORY = []
    BEST_AUC = -1.0


def fit_config(server_round: int) -> Dict[str, float]:
    return {
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
    }


def weighted_average(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    if not metrics:
        return {}
    total_examples = sum(num_examples for num_examples, _ in metrics)
    metric_names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    aggregated = {}
    for name in metric_names:
        aggregated[name] = sum(num_examples * values.get(name, 0.0) for num_examples, values in metrics) / max(total_examples, 1)
    return aggregated


def get_server_evaluate_fn():
    global BEST_AUC
    metadata = read_metadata()
    x_test, y_test = load_global_test_data()
    model = build_model(input_dim=int(metadata["input_dim"]), learning_rate=LEARNING_RATE)

    def evaluate(server_round: int, parameters, config):
        global BEST_AUC
        set_model_weights(model, parameters)
        loss = float(model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0))
        y_prob = model.predict(x_test, batch_size=BATCH_SIZE, verbose=0).ravel()
        metric_values, cm, fpr, tpr = compute_classification_metrics(y_test, y_prob)
        round_record = {"round": int(server_round), "loss": loss, **metric_values}
        ROUND_HISTORY.append(round_record)
        save_round_history(history=ROUND_HISTORY, json_path=ROUND_METRICS_JSON, csv_path=ROUND_METRICS_CSV, final_metrics_path=FINAL_METRICS_JSON)
        save_confusion_matrix(cm=cm, output_path=CONFUSION_MATRIX_PATH, title=f"Federated Fraud Detection Confusion Matrix - Round {server_round}")
        save_roc_curve(fpr=fpr, tpr=tpr, roc_auc=metric_values["roc_auc"], output_path=ROC_CURVE_PATH, title=f"Federated Fraud Detection ROC Curve - Round {server_round}")
        if metric_values["roc_auc"] >= BEST_AUC:
            BEST_AUC = metric_values["roc_auc"]
            model.save(MODEL_PATH)
        return loss, metric_values

    return evaluate


def server_fn(context: Context):
    ensure_directories()
    set_global_seed()
    metadata = read_metadata()
    model = build_model(input_dim=int(metadata["input_dim"]), learning_rate=LEARNING_RATE)
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=ndarrays_to_parameters(model.get_weights()),
        on_fit_config_fn=fit_config,
        evaluate_fn=get_server_evaluate_fn(),
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    config = ServerConfig(num_rounds=NUM_ROUNDS)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
