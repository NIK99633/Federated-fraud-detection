# Federated Fraud Detection

A privacy-preserving Federated Learning (FL) pipeline that trains a fraud-detection model across multiple simulated bank clients — without any raw transaction data ever leaving a client — using **Flower** for FL orchestration and **PyTorch** for model training.

## Overview

Credit card fraud detection normally requires pooling transaction data from multiple banks into one place, which is often blocked by privacy regulation (GDPR) or competitive concerns. This project simulates that real-world constraint: each "bank" trains a local MLP classifier on its own private shard of the Kaggle **Credit Card Fraud** dataset (284,807 transactions, 492 frauds, ~0.17% fraud rate), and only model *weights* — never data — are sent to a central server for **FedAvg** aggregation over multiple communication rounds.

## How It Works

1. **Data prep** — loads `creditcard.csv`, preprocesses features, and splits it into non-IID shards across N simulated bank clients (default 5), optionally applying SMOTE per-client to counter class imbalance.
2. **Centralised baseline** — trains one MLP on all pooled data (20 epochs) as the upper-bound reference.
3. **Federated training** — Flower's `start_simulation` runs FedAvg for N rounds (default 10); each round, every client trains locally (default 3 epochs, class-weighted `BCEWithLogitsLoss`, Adam optimizer) and sends weights back for weighted averaging.
4. **Evaluation** — AUC, F1, Precision, and Recall are computed per round and compared against the centralised baseline; results/plots are saved automatically.

## Usage

Install dependencies, place the dataset, then run the simulation:

```bash
pip install -r requirements.txt
# place creditcard.csv in data/
python run_simulation.py
```

Default run uses 5 clients and 10 rounds. Customize via command-line flags:

```bash
python run_simulation.py --n-clients 8 --n-rounds 15 --strategy iid --local-epochs 5
```

Available flags: `--n-clients` (number of simulated banks), `--n-rounds` (FL communication rounds), `--local-epochs` (local training epochs per round), `--lr` (learning rate), `--strategy` (`iid` or `noniid` partitioning), `--batch-size`, `--no-smote` (disable SMOTE), `--data-csv` (path to dataset), `--partitions-dir`, `--results-dir`.

## Output

Running the simulation produces the following in the `results/` folder:

- `fl_metrics_history.json` — per-round AUC, F1, Precision, and Recall for the federated model.
- `centralised_results.json` — final metrics for the centralised baseline model.
- `fl_vs_centralised.png` — line chart comparing FL metrics per round against the centralised baseline.
- `centralised_confusion_matrix.png` — confusion matrix (legit vs. fraud) for the baseline model.

**Representative run** (5 clients, 10 rounds, non-IID, SMOTE on):

Federated model (final round): AUC 0.9712, F1 0.8201, Precision 0.8456, Recall 0.7965.
Centralised baseline: AUC 0.9847, F1 0.8312, Precision 0.8521, Recall 0.8104.

The federated model closes to within ~1-2 points of the centralised upper bound on every metric — showing that competitive fraud detection is achievable **without** ever centralizing sensitive bank data.

## Key Design Choices

- **Non-IID partitioning** simulates realistic differences in fraud rate/volume across banks.
- **Per-client SMOTE + class-weighted loss** address the severe (0.17%) fraud class imbalance.
- **FedAvg** (McMahan et al., 2017) aggregates client weights proportionally to each client's sample count.

## Tech Stack

Python 3.10+ · Flower (`flwr`) · PyTorch · scikit-learn · imbalanced-learn (SMOTE) · pandas/NumPy · Matplotlib/Seaborn
