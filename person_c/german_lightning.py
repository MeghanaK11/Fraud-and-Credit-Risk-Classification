
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data import (
    load_german, GERMAN_TARGET,
    GERMAN_CATEGORICAL, GERMAN_COST_FN, GERMAN_COST_FP,
    german_cv_splits
)
from common.metrics import probability_metrics

OUT = ROOT / "results" / "person_c"
OUT.mkdir(parents=True, exist_ok=True)

L.seed_everything(42, workers=True)
torch.set_num_threads(2)


class CreditModel(L.LightningModule):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.network(x).squeeze(1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log("train_loss", loss, prog_bar=False)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=0.001, weight_decay=0.001
        )


def prepare_data(X_train, X_test):
    train = X_train.copy()
    test = X_test.copy()

    for col in GERMAN_CATEGORICAL:
        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)

    train = pd.get_dummies(train, columns=GERMAN_CATEGORICAL, dtype=float)
    test = pd.get_dummies(test, columns=GERMAN_CATEGORICAL, dtype=float)

    test = test.reindex(columns=train.columns, fill_value=0)

    scaler = StandardScaler()
    train_arr = scaler.fit_transform(train).astype("float32")
    test_arr = scaler.transform(test).astype("float32")

    return train_arr, test_arr


def predict_model(X_train, y_train, X_test, seed):
    L.seed_everything(seed, workers=True)

    model = CreditModel(X_train.shape[1])

    dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )

    loader = DataLoader(
        dataset, batch_size=64, shuffle=True,
        num_workers=0
    )

    trainer = L.Trainer(
        max_epochs=35,
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
    train_time = time.perf_counter() - start

    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(
            model(torch.tensor(X_test, dtype=torch.float32))
        ).numpy()

    return probabilities, train_time


def cost(y, probabilities, threshold):
    predicted_bad = probabilities >= threshold

    false_negative = ((y == 1) & (~predicted_bad)).sum()
    false_positive = ((y == 0) & predicted_bad).sum()

    return (
        GERMAN_COST_FN * false_negative
        + GERMAN_COST_FP * false_positive
    ) / len(y)


def main():
    print("\nLoading German Credit dataset...")
    df = load_german()

    X = df.drop(columns=GERMAN_TARGET)
    y = df[GERMAN_TARGET].to_numpy()

    print("Total applicants:", len(df))
    print("Bad applicants:", y.sum())
    print("Good applicants:", (y == 0).sum())

    rows = []
    fairness_rows = []

    # Default seed, so the folds match XGBoost, fastai, Keras and TabNet.
    for repeat, fold, train_idx, test_idx in german_cv_splits(df):
        seed = 42 + repeat * 5 + fold

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        X_train_arr, X_test_arr = prepare_data(X_train, X_test)

        probabilities, train_time = predict_model(
            X_train_arr, y_train, X_test_arr, seed
        )

        pred_default = probabilities >= 0.5
        pred_theory = probabilities >= (1 / 6)

        rows.append({
            "repeat": repeat,
            "fold": fold,
            "seed": seed,
            "roc_auc": roc_auc_score(y_test, probabilities),
            "pr_auc": average_precision_score(y_test, probabilities),
            "brier": brier_score_loss(y_test, probabilities),
            # The brief's regression metrics, on predicted probabilities
            # (same definitions as common.metrics.probability_metrics).
            **{k: v for k, v in probability_metrics(y_test, probabilities).items()
               if k != "brier"},
            "cost_default": cost(y_test, probabilities, 0.5),
            "cost_theory": cost(y_test, probabilities, 1 / 6),
            "cost_reject_all": cost(
                y_test, np.ones(len(y_test)), 0.5
            ),
            "train_seconds": train_time,
            "parameters": sum(
                p.numel() for p in CreditModel(X_train_arr.shape[1]).parameters()
            )
        })

        test_groups = df.iloc[test_idx].copy()
        test_groups["actual_bad"] = y_test
        test_groups["predicted_bad"] = pred_theory
        test_groups["probability_bad"] = probabilities

        test_groups["age_group"] = pd.cut(
            test_groups["age"],
            bins=[0, 25, 35, 50, 100],
            labels=["18-25", "26-35", "36-50", "51+"]
        )

        for column in ["age_group", "personal_status_sex"]:
            for group, subset in test_groups.groupby(
                column, observed=True
            ):
                if len(subset) == 0:
                    continue

                actual = subset["actual_bad"].to_numpy()
                predicted = subset["predicted_bad"].to_numpy()

                fairness_rows.append({
                    "repeat": repeat,
                    "fold": fold,
                    "attribute": column,
                    "group": str(group),
                    "applicants": len(subset),
                    "approval_rate": float((~predicted).mean()),
                    "bad_missed_rate": float(
                        ((actual == 1) & (~predicted)).sum()
                        / max((actual == 1).sum(), 1)
                    ),
                    "average_cost": cost(
                        actual,
                        subset["probability_bad"].to_numpy(),
                        1 / 6
                    )
                })

        print(
            f"Repeat {repeat+1}/5 | Fold {fold+1}/5 | "
            f"ROC-AUC: {rows[-1]['roc_auc']:.3f} | "
            f"Cost: {rows[-1]['cost_theory']:.3f}"
        )

    results = pd.DataFrame(rows)
    results.to_csv(OUT / "lightning_german_runs.csv", index=False)

    summary = results.groupby("repeat").agg({
        "roc_auc": "mean",
        "pr_auc": "mean",
        "brier": "mean",
        "cost_default": "mean",
        "cost_theory": "mean",
        "cost_reject_all": "mean",
        "train_seconds": "mean"
    })

    summary.to_csv(OUT / "lightning_german_repeat_summary.csv")

    fairness = pd.DataFrame(fairness_rows)
    fairness.to_csv(OUT / "lightning_fairness.csv", index=False)

    print("\n========== LIGHTNING RESULTS ==========")
    print(results[
        ["roc_auc", "pr_auc", "brier", "cost_default",
         "cost_theory", "train_seconds"]
    ].agg(["mean", "std"]).round(4))

    print("\n========== FAIRNESS RESULTS ==========")
    print(
        fairness.groupby(["attribute", "group"])[
            ["applicants", "approval_rate", "bad_missed_rate",
             "average_cost"]
        ].mean().round(3).to_string()
    )

    print("\nResults saved in:", OUT)
    print("25 cross-validation runs completed.")


if __name__ == "__main__":
    main()
