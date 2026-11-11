"""Collect every framework's results into one table.

Reads the per-run CSVs written by each person's code and writes
results/summary_all_frameworks.csv (mean and std per framework).
Run it after everyone's experiments have finished.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"


def fraud_runs():
    a = pd.read_csv(RES / "person_a" / "fraud_runs.csv")
    a = a[(a.split == "temporal") & (a.features == "base")]
    parts = [
        a[a.model == "xgboost"].assign(framework="XGBoost"),
        a[a.model == "fastai"].assign(framework="fastai"),
        pd.read_csv(RES / "person_b" / "keras_fraud_runs.csv").assign(framework="Keras"),
        pd.read_csv(RES / "person_b" / "tabnet_fraud_runs.csv").assign(framework="TabNet"),
        pd.read_csv(RES / "person_c" / "lightning_fraud_runs.csv").assign(framework="Lightning"),
    ]
    cols = ["framework", "pr_auc", "roc_auc", "recall_at_fpr_0.001", "brier"]
    return pd.concat([p[cols] for p in parts], ignore_index=True)


def german_runs():
    a = pd.read_csv(RES / "person_a" / "german_runs.csv").rename(
        columns={"cost_default": "cost_0_5", "cost_theory": "cost_1_6"})
    keras = pd.read_csv(RES / "person_b" / "keras_german_25fold_metrics.csv")
    tabnet = pd.read_csv(RES / "person_b" / "tabnet_german_25fold_metrics.csv")
    light = pd.read_csv(RES / "person_c" / "lightning_german_runs.csv")
    b_names = {"cost_05": "cost_0_5", "cost_0167": "cost_1_6"}
    parts = [
        a[a.model == "xgboost"].assign(framework="XGBoost"),
        a[a.model == "fastai"].assign(framework="fastai"),
        keras.rename(columns=b_names).assign(framework="Keras"),
        tabnet.rename(columns=b_names).assign(framework="TabNet"),
        light.rename(columns={"cost_default": "cost_0_5", "cost_theory": "cost_1_6"})
             .assign(framework="Lightning"),
    ]
    cols = ["framework", "roc_auc", "pr_auc", "brier", "cost_0_5", "cost_1_6"]
    return pd.concat([p[cols] for p in parts], ignore_index=True)


def main():
    order = ["XGBoost", "fastai", "Keras", "TabNet", "Lightning"]
    f = fraud_runs()
    g = german_runs()
    fs = f.groupby("framework").agg(["mean", "std"]).reindex(order)
    fs.columns = [f"fraud_{m}_{s}" for m, s in fs.columns]
    fs["fraud_runs"] = f.groupby("framework").size().reindex(order)
    gs = g.groupby("framework").agg(["mean", "std"]).reindex(order)
    gs.columns = [f"german_{m}_{s}" for m, s in gs.columns]
    gs["german_folds"] = g.groupby("framework").size().reindex(order)
    out = fs.join(gs)
    out.to_csv(RES / "summary_all_frameworks.csv")
    show = ["fraud_runs", "fraud_pr_auc_mean", "fraud_pr_auc_std", "fraud_roc_auc_mean",
            "fraud_recall_at_fpr_0.001_mean", "german_folds", "german_roc_auc_mean",
            "german_brier_mean", "german_cost_0_5_mean", "german_cost_1_6_mean"]
    print(out[show].round(3).to_string())


if __name__ == "__main__":
    main()
