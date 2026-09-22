import argparse
import os
import traceback
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Federated Fraud Detection system with Flower simulation.")
    parser.add_argument("--data-path", type=str, default="data/creditcard.csv", help="Path to Kaggle creditcard.csv")
    parser.add_argument("--num-clients", type=int, default=4, help="Number of simulated banks. Fixed to 4 for this project.")
    parser.add_argument("--num-rounds", type=int, default=10, help="Number of federated rounds.")
    parser.add_argument("--local-epochs", type=int, default=2, help="Local epochs per federated round.")
    parser.add_argument("--batch-size", type=int, default=256, help="Mini-batch size for local client training.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate for the Keras optimizer.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Global hold-out test set ratio.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--client-cpus", type=float, default=1.0, help="CPU resources per simulated Flower client.")
    parser.add_argument("--client-gpus", type=float, default=0.0, help="GPU fraction per simulated Flower client.")
    parser.add_argument("--enable-tf-gpu-growth", action="store_true", help="Enable TensorFlow GPU memory growth in Flower simulation.")
    parser.add_argument("--recreate-data", action="store_true", help="Force regeneration of saved partitions and metadata.")
    return parser.parse_args()


def export_runtime_config(args) -> None:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["FFD_DATASET_PATH"] = str(Path(args.data_path).resolve())
    os.environ["FFD_NUM_CLIENTS"] = str(args.num_clients)
    os.environ["FFD_NUM_ROUNDS"] = str(args.num_rounds)
    os.environ["FFD_LOCAL_EPOCHS"] = str(args.local_epochs)
    os.environ["FFD_BATCH_SIZE"] = str(args.batch_size)
    os.environ["FFD_LEARNING_RATE"] = str(args.learning_rate)
    os.environ["FFD_TEST_SIZE"] = str(args.test_size)
    os.environ["FFD_RANDOM_STATE"] = str(args.random_state)


def main() -> None:
    args = parse_args()
    if args.num_clients != 4:
        raise ValueError("This academic submission expects exactly 4 simulated banks.")
    export_runtime_config(args)
    try:
        from flwr.simulation import run_simulation
        from app.config import CONFUSION_MATRIX_PATH, FINAL_METRICS_JSON, MODEL_PATH, ROC_CURVE_PATH, ROUND_METRICS_CSV, ROUND_METRICS_JSON, ensure_directories
        from app.dataset import prepare_federated_data
        from app.client_app import app as client_app
        from app.server_app import app as server_app, reset_server_state

        ensure_directories()
        reset_server_state()
        metadata = prepare_federated_data(force_recreate=args.recreate_data)

        print("=" * 72)
        print("FEDERATED FRAUD DETECTION SIMULATION")
        print("=" * 72)
        print(f"Dataset      : {Path(args.data_path).resolve()}")
        print(f"Clients      : {args.num_clients}")
        print(f"Rounds       : {args.num_rounds}")
        print(f"Local epochs : {args.local_epochs}")
        print(f"Input dim    : {metadata['input_dim']}")
        print("=" * 72)

        run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=args.num_clients,
            backend_name="ray",
            backend_config={
                "client_resources": {
                    "num_cpus": float(args.client_cpus),
                    "num_gpus": float(args.client_gpus),
                }
            },
            enable_tf_gpu_growth=args.enable_tf_gpu_growth,
        )

        print("\nTraining completed successfully.")
        print(f"Saved model            : {MODEL_PATH}")
        print(f"Round metrics JSON     : {ROUND_METRICS_JSON}")
        print(f"Round metrics CSV      : {ROUND_METRICS_CSV}")
        print(f"Final metrics JSON     : {FINAL_METRICS_JSON}")
        print(f"Confusion matrix image : {CONFUSION_MATRIX_PATH}")
        print(f"ROC curve image        : {ROC_CURVE_PATH}")

    except Exception as exc:
        print("\n[ERROR] Simulation failed.")
        print(str(exc))
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
