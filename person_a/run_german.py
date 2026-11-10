"""Person A experiments on German Credit.

Repeated stratified 5-fold CV (5 repeats = 25 folds) for XGBoost and fastai.
Inside each training fold, 20% is held out (stratified) for early stopping
and for tuning a cost-minimising threshold.

Three threshold policies are compared on every test fold:
  default  : predict bad when P(bad) >= 0.5
  theory   : predict bad when P(bad) >= 1 / (1 + 5) = 0.167, the Bayes
             threshold for the 5:1 cost matrix if probabilities are calibrated
  tuned    : threshold that minimised cost on the inner validation split
Reference policy: reject everyone, which costs 0.70 per applicant here.

Writes results/person_a/german_runs.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data import (  # noqa: E402
    GERMAN_CATEGORICAL, GERMAN_COST_FN, GERMAN_COST_FP, GERMAN_TARGET,
    german_cv_splits, load_german,
)
from common.metrics import all_metrics, best_threshold, german_cost  # noqa: E402
from person_a.models import fastai_fit_predict, xgb_fit_predict  # noqa: E402

OUT = ROOT / "results" / "person_a"
THEORY_T = GERMAN_COST_FP / (GERMAN_COST_FP + GERMAN_COST_FN)


def cost(y, flagged):
    return german_cost(y, flagged, GERMAN_COST_FN, GERMAN_COST_FP)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_german()
    X_all = df.drop(columns=GERMAN_TARGET)
    y_all = df[GERMAN_TARGET].to_numpy()
    reject_all = cost(y_all, np.ones_like(y_all))
    print(f"rows {len(df)}, bad {y_all.sum()}, reject-everyone cost per applicant {reject_all:.3f}")

    models = {
        "xgboost": lambda *a, **k: xgb_fit_predict(*a, **k, cat_cols=GERMAN_CATEGORICAL),
        "fastai": lambda *a, **k: fastai_fit_predict(*a, **k, cat_cols=GERMAN_CATEGORICAL,
                                                     epochs=10, bs=64),
    }
    rows = []
    for rep, fold, tr, te in german_cv_splits(df):
        seed = rep * 5 + fold
        tr_in, va = train_test_split(tr, test_size=0.2, stratify=y_all[tr], random_state=seed)
        y_te = y_all[te]
        for mname, fit in models.items():
            p_va, p_te, train_s, _ = fit(
                X_all.iloc[tr_in], y_all[tr_in], X_all.iloc[va], y_all[va], X_all.iloc[te], seed=seed)
            m = all_metrics(y_te, p_te, fpr_targets=(0.1,))
            t_tuned, _ = best_threshold(y_all[va], p_va, cost)
            for policy, t in (("default", 0.5), ("theory", THEORY_T), ("tuned", t_tuned)):
                flagged = p_te >= t
                m[f"cost_{policy}"] = cost(y_te, flagged)
                m[f"threshold_{policy}"] = t
                m[f"recall_bad_{policy}"] = float(flagged[y_te == 1].mean())
                m[f"reject_rate_{policy}"] = float(flagged.mean())
            m["cost_reject_all"] = cost(y_te, np.ones_like(y_te))
            rows.append(dict(repeat=rep, fold=fold, model=mname, train_seconds=train_s, **m))
            print(f"rep {rep} fold {fold} {mname:8s} ROC-AUC {m['roc_auc']:.3f} "
                  f"cost default {m['cost_default']:.3f} theory {m['cost_theory']:.3f} "
                  f"tuned {m['cost_tuned']:.3f}")

    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "german_runs.csv", index=False)
    cols = ["roc_auc", "pr_auc", "brier", "rmse", "mae", "r2",
            "cost_default", "cost_theory", "cost_tuned", "cost_reject_all"]
    print(runs.groupby("model")[cols].agg(["mean", "std"]).round(3).T.to_string())


if __name__ == "__main__":
    main()
