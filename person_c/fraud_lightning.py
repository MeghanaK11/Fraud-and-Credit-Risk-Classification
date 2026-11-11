
import sys
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import lightning as L

from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data import FRAUD_TARGET, load_fraud, temporal_split
from common.metrics import all_metrics
from person_a.features import build_features

OUT = ROOT / "results" / "person_c"
OUT.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(2)


class FraudModel(L.LightningModule):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.network(x).squeeze(1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


def main():
    df = load_fraud()
    y = df[FRAUD_TARGET].to_numpy()

    print("Transactions:", len(df))
    print("Fraud cases:", y.sum())
    print("Duplicates removed:", df.attrs["n_duplicates_dropped"])

    X = build_features(df, with_velocity=False)

    train_idx, val_idx, test_idx = temporal_split(df)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X.iloc[train_idx]).astype("float32")
    X_test = scaler.transform(X.iloc[test_idx]).astype("float32")

    y_train = y[train_idx]
    y_test = y[test_idx]

    dataset = TensorDataset(
        torch.tensor(X_train),
        torch.tensor(y_train, dtype=torch.float32)
    )

    loader = DataLoader(
        dataset, batch_size=2048, shuffle=True, num_workers=0
    )

    rows = []

    for seed in range(10):
        print(f"\nTraining seed {seed + 1}/10")
        L.seed_everything(seed, workers=True)

        model = FraudModel(X_train.shape[1])

        trainer = L.Trainer(
            max_epochs=5,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            deterministic=True
        )

        start = time.perf_counter()
        trainer.fit(model, train_dataloaders=loader)
        train_seconds = time.perf_counter() - start

        model.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(
                model(torch.tensor(X_test))
            ).numpy()

        metrics = all_metrics(y_test, probabilities)

        rows.append({
            "seed": seed,
            "train_seconds": train_seconds,
            "test_transactions": len(y_test),
            "test_frauds": int(y_test.sum()),
            **metrics
        })

        print(
            f"PR-AUC: {metrics['pr_auc']:.4f} | "
            f"ROC-AUC: {metrics['roc_auc']:.4f} | "
            f"RMSE: {metrics['rmse']:.4f}"
        )

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "lightning_fraud_runs.csv", index=False)

    print("\n====== LIGHTNING FRAUD SUMMARY ======")
    print(
        results[
            ["pr_auc", "roc_auc", "rmse", "mae",
             "r2", "train_seconds"]
        ].agg(["mean", "std"]).round(4)
    )

    print("\nSaved:", OUT / "lightning_fraud_runs.csv")


if __name__ == "__main__":
    main()
