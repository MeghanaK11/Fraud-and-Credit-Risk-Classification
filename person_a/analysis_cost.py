"""Cost-based evaluation of the temporal-split fraud models.

Cost model (see common.metrics.fraud_cost):
  - every alert costs a fixed review fee R
  - every missed fraud costs its transaction amount
  - card-testing variant: a missed probe (fraud amount <= 1.00) also lets
    a follow-on fraud through. Its size L is estimated from the training
    period as the mean amount of frauds above 1.00. Nothing here is typed
    in by hand except the grid of review fees, which is swept because we
    do not know a bank's real review cost.

For every run the threshold is picked on the validation period (the last
15% of the training time range) and then applied to the test period.

Writes
  results/person_a/cost_runs.csv     one row per run x scenario
  results/person_a/cost_summary.csv  mean/std per model, features, scenario
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data import FRAUD_TARGET, load_fraud, temporal_split  # noqa: E402
from common.metrics import best_threshold, fraud_cost  # noqa: E402
from person_a.features import MICRO_AMOUNT  # noqa: E402

OUT = ROOT / "results" / "person_a"
REVIEW_FEES = (1, 2, 5, 10)
SEEDS = range(10)


def follow_on_loss(df, train_idx):
    y = df[FRAUD_TARGET].to_numpy()
    amt = df["Amount"].to_numpy()
    sel = (y[train_idx] == 1) & (amt[train_idx] > MICRO_AMOUNT)
    return float(amt[train_idx][sel].mean())


def main():
    df = load_fraud()
    y = df[FRAUD_TARGET].to_numpy()
    amt = df["Amount"].to_numpy()
    tr, va, te = temporal_split(df)
    L = follow_on_loss(df, np.r_[tr, va])
    probes_te = (y[te] == 1) & (amt[te] <= MICRO_AMOUNT)
    print(f"follow-on loss L (mean training fraud > {MICRO_AMOUNT:.2f}): {L:.2f}")
    print(f"test frauds {y[te].sum()}, of which probes {probes_te.sum()}, "
          f"total fraud amount in test {amt[te][y[te] == 1].sum():.2f}")

    P = np.load(OUT / "fraud_predictions.npz")
    rows = []
    for R in REVIEW_FEES:
        for card_testing in (False, True):
            tl = L if card_testing else None
            scen = f"R={R}" + (" +card-testing" if card_testing else "")

            def cost_va(yy, flag):
                return fraud_cost(yy, flag, amt[va], R, tl)

            def cost_te(yy, flag):
                return fraud_cost(yy, flag, amt[te], R, tl)

            none = cost_te(y[te], np.zeros(len(te), bool))
            rows.append(dict(scenario=scen, review_fee=R, card_testing=card_testing,
                             model="flag nothing", features="-", seed=-1, cost=none))
            for fname in ("base", "velocity"):
                for mname in ("xgboost", "fastai"):
                    for s in SEEDS:
                        key = f"temporal_{fname}_{mname}_{s}"
                        p_va, p_te = P[f"pva_{key}"], P[f"pte_{key}"]
                        t, _ = best_threshold(y[va], p_va, cost_va)
                        flag = p_te >= t
                        frauds = y[te] == 1
                        rows.append(dict(
                            scenario=scen, review_fee=R, card_testing=card_testing,
                            model=mname, features=fname, seed=s,
                            threshold=t,
                            cost=cost_te(y[te], flag),
                            cost_at_0_5=cost_te(y[te], p_te >= 0.5),
                            cost_oracle=best_threshold(y[te], p_te, cost_te)[1],
                            alerts=int(flag.sum()),
                            recall=float(flag[frauds].mean()),
                            recall_probes=float(flag[probes_te].mean()),
                            recall_large=float(flag[frauds & ~probes_te].mean()),
                            saving_vs_nothing=1 - cost_te(y[te], flag) / none,
                        ))
    runs = pd.DataFrame(rows)
    runs.to_csv(OUT / "cost_runs.csv", index=False)

    m = runs[runs.seed >= 0]
    summary = (m.groupby(["scenario", "review_fee", "card_testing", "features", "model"])
               [["cost", "cost_at_0_5", "cost_oracle", "alerts", "recall",
                 "recall_probes", "recall_large", "saving_vs_nothing", "threshold"]]
               .agg(["mean", "std"]))
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    base = runs[runs.seed < 0][["scenario", "cost"]].rename(columns={"cost": "cost_flag_nothing"})
    summary = summary.merge(base, on="scenario")
    summary["follow_on_loss_L"] = L
    summary.to_csv(OUT / "cost_summary.csv", index=False)
    show = ["scenario", "features", "model", "cost_mean", "cost_std", "cost_flag_nothing",
            "recall_probes_mean", "recall_large_mean", "alerts_mean"]
    print(summary[show].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
