"""Does a random split let the model memorise the hour-11 attack?

Hour 11 holds a burst of frauds a few seconds apart. Under a random split
some of those rows land in training and some in test. If the model is
recognising this one attack instead of learning fraud in general, its
recall on the attack frauds in test should drop once their siblings are
removed from training.

For each of 10 random splits (same splits as run_fraud.py):
  with_siblings    : train as usual
  without_siblings : drop the hour-11 fraud rows from train and validation
Recall is measured at the threshold giving 0.1% FPR over test normals.

Writes results/person_a/attack_control.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.data import FRAUD_TARGET, load_fraud, random_split  # noqa: E402
from person_a.features import build_features  # noqa: E402
from person_a.models import MODELS  # noqa: E402
from person_a.run_fraud import _recall_subset  # noqa: E402

OUT = ROOT / "results" / "person_a"


def main():
    df = load_fraud()
    y = df[FRAUD_TARGET].to_numpy()
    hour = (df["Time"] // 3600).astype(int).to_numpy()
    attack = (hour == 11) & (y == 1)
    X = build_features(df, False)
    rows = []
    for seed in range(10):
        tr, va, te = random_split(df, seed)
        for variant in ("with_siblings", "without_siblings"):
            tr_v, va_v = tr, va
            if variant == "without_siblings":
                tr_v, va_v = tr[~attack[tr]], va[~attack[va]]
            for mname, fit in MODELS.items():
                _, p_te, _, _ = fit(X.iloc[tr_v], y[tr_v], X.iloc[va_v], y[va_v], X.iloc[te], seed=seed)
                in_attack = hour[te] == 11
                rows.append(dict(
                    seed=seed, variant=variant, model=mname,
                    attack_in_train=int(attack[tr_v].sum()),
                    attack_in_test=int(attack[te].sum()),
                    recall_attack=_recall_subset(y[te], p_te, in_attack),
                    recall_other=_recall_subset(y[te], p_te, ~in_attack),
                ))
                print(rows[-1])
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "attack_control.csv", index=False)
    print(out.groupby(["model", "variant"])[["recall_attack", "recall_other"]]
          .agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()
