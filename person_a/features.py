"""Feature sets for the credit card data.

The sample has no card or customer ID, so velocity here means activity
across the whole system in the recent past, not per card. Every feature
only looks at rows that came earlier in time order, and never at labels,
so it could be computed live when a transaction arrives.
"""
import numpy as np
import pandas as pd

from common.data import PCA_COLS

WINDOWS = (60, 300)        # seconds
MICRO_AMOUNT = 1.0         # card-testing probes in this data are mostly <= 1.00


def base_features(df):
    """V1..V28 plus log amount. Raw Time is left out: under a temporal split
    the test period has Time values the model never saw in training."""
    X = df[PCA_COLS].copy()
    X["log_amount"] = np.log1p(df["Amount"])
    return X


def velocity_features(df):
    """Causal system-level velocity features (needs df sorted by Time)."""
    t = df["Time"].to_numpy(dtype=float)
    amt = df["Amount"].to_numpy(dtype=float)
    micro = (amt <= MICRO_AMOUNT).astype(float)
    n = len(df)
    pos = np.arange(n)
    cs_micro = np.concatenate([[0.0], np.cumsum(micro)])
    cs_amt = np.concatenate([[0.0], np.cumsum(amt)])

    out = {}
    for w in WINDOWS:
        left = np.searchsorted(t, t - w, side="left")
        cnt = pos - left                                # earlier rows inside the window
        out[f"txn_last_{w}s"] = cnt
        out[f"micro_last_{w}s"] = cs_micro[pos] - cs_micro[left]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_amt = (cs_amt[pos] - cs_amt[left]) / cnt
        out[f"mean_amt_last_{w}s"] = np.nan_to_num(mean_amt, nan=0.0)
    # Share of micro transactions in the last 300 s, a direct burst signal.
    with np.errstate(invalid="ignore", divide="ignore"):
        out["micro_share_last_300s"] = np.nan_to_num(
            out["micro_last_300s"] / out["txn_last_300s"], nan=0.0)
    gap = np.diff(t, prepend=t[0])
    out["secs_since_prev"] = gap
    return pd.DataFrame(out, index=df.index)


def build_features(df, with_velocity):
    X = base_features(df)
    if with_velocity:
        X = pd.concat([X, velocity_features(df)], axis=1)
    return X.astype("float32")
