"""Person A experiments on the credit card sample.

Grid: model (xgboost, fastai) x features (base, base+velocity)
      x split (temporal, random) x 10 seeds.

Temporal split: the split is fixed, so the 10 seeds only measure model
randomness. Random split: seed k gives a different stratified split and
model seed, so the spread covers both.

Writes
  results/person_a/fraud_runs.csv         one row per run, test metrics
  results/person_a/fraud_ci.csv           bootstrap 95% CIs (temporal, seed-averaged)
  results/person_a/fraud_predictions.npz  val/test probabilities for cost analysis
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data import FRAUD_TARGET, load_fraud, random_split, temporal_split  # noqa: E402
from common.metrics import all_metrics, bootstrap_ci, recall_at_fpr  # noqa: E402
from person_a.features import build_features  # noqa: E402
from person_a.models import MODELS  # noqa: E402

OUT = ROOT / "results" / "person_a"
SEEDS = range(10)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_fraud()
    y = df[FRAUD_TARGET].to_numpy()
    hour = (df["Time"] // 3600).astype(int).to_numpy()
    print(f"rows after dedupe: {len(df)} (dropped {df.attrs['n_duplicates_dropped']}), frauds: {y.sum()}")

    feats = {"base": build_features(df, False), "velocity": build_features(df, True)}
    rows, preds = [], {}

    for split in ("temporal", "random"):
        for seed in SEEDS:
            tr, va, te = temporal_split(df) if split == "temporal" else random_split(df, seed)
            preds[f"val_idx_{split}_{seed}"] = va
            preds[f"test_idx_{split}_{seed}"] = te
            for fname, X in feats.items():
                for mname, fit in MODELS.items():
                    p_va, p_te, train_s, _ = fit(
                        X.iloc[tr], y[tr], X.iloc[va], y[va], X.iloc[te], seed=seed)
                    m = all_metrics(y[te], p_te)
                    # Recall on the hour-11 attack versus every other fraud in the
                    # test set, at the threshold giving 0.1% FPR on the test set.
                    attack = hour[te] == 11
                    m["test_frauds"] = int(y[te].sum())
                    m["test_attack_frauds"] = int(y[te][attack].sum())
                    m["recall_attack"] = _recall_subset(y[te], p_te, attack)
                    m["recall_non_attack"] = _recall_subset(y[te], p_te, ~attack)
                    rows.append(dict(split=split, seed=seed, features=fname, model=mname,
                                     train_seconds=train_s, **m))
                    key = f"{split}_{fname}_{mname}_{seed}"
                    preds[f"pva_{key}"] = p_va.astype("float32")
                    preds[f"pte_{key}"] = p_te.astype("float32")
                    print(f"{split:8s} seed {seed} {fname:8s} {mname:8s} "
                          f"PR-AUC {m['pr_auc']:.3f}  ROC-AUC {m['roc_auc']:.3f}  {train_s:.1f}s")

    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "fraud_runs.csv", index=False)
    np.savez_compressed(OUT / "fraud_predictions.npz", **preds)

    # Bootstrap CIs on the temporal test set, using the seed-averaged prediction.
    tr, va, te = temporal_split(df)
    ci_rows = []
    for fname in feats:
        for mname in MODELS:
            p = np.mean([preds[f"pte_temporal_{fname}_{mname}_{s}"] for s in SEEDS], axis=0)
            for metric, fn in [("pr_auc", average_precision_score),
                               ("recall_at_fpr_0.001", lambda a, b: recall_at_fpr(a, b, 0.001))]:
                lo, hi = bootstrap_ci(y[te], p, fn, n_boot=1000, seed=0)
                ci_rows.append(dict(features=fname, model=mname, metric=metric,
                                    point=fn(y[te], p), ci_low=lo, ci_high=hi))
    pd.DataFrame(ci_rows).to_csv(OUT / "fraud_ci.csv", index=False)
    print(pd.DataFrame(ci_rows).round(3).to_string(index=False))


def _recall_subset(y, p, mask, max_fpr=0.001):
    """Recall on frauds inside mask, threshold set for max_fpr over all normals."""
    neg = np.sort(p[y == 0])[::-1]
    k = int(np.floor(max_fpr * len(neg)))
    thr = neg[k] if k < len(neg) else 0.0  # flag scores strictly above the (k+1)-th normal
    pos = (y == 1) & mask
    return float((p[pos] > thr).mean()) if pos.any() else np.nan


if __name__ == "__main__":
    main()
