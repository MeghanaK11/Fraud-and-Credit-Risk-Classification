"""Shared data loading and splitting for both datasets.

Every team member should import from here so that all frameworks are
trained and tested on exactly the same rows.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Credit card sample

FRAUD_TARGET = "Class"
PCA_COLS = [f"V{i}" for i in range(1, 29)]


def load_fraud(path=None):
    """Load the credit card sample, drop exact duplicate rows, keep time order."""
    path = path or DATA_DIR / "creditcard_sample.csv"
    df = pd.read_csv(path)
    n_before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    df.attrs["n_duplicates_dropped"] = n_before - len(df)
    # The file is already sorted by Time; sort again in case a teammate
    # passes a shuffled copy. A stable sort keeps file order inside a second.
    df = df.sort_values("Time", kind="stable").reset_index(drop=True)
    return df


def temporal_split(df, test_frac=0.30, val_frac=0.15):
    """Earliest rows train, latest rows test.

    The validation set is the last val_frac of the training period, so
    thresholds and early stopping are also chosen on data from the past.
    Returns index arrays (train, val, test).
    """
    n = len(df)
    test_start = int(n * (1 - test_frac))
    val_start = int(test_start * (1 - val_frac))
    idx = np.arange(n)
    return idx[:val_start], idx[val_start:test_start], idx[test_start:]


def random_split(df, seed, test_frac=0.30, val_frac=0.15):
    """Stratified random split with the same sizes as temporal_split."""
    idx = np.arange(len(df))
    y = df[FRAUD_TARGET].values
    trval, test = train_test_split(idx, test_size=test_frac, stratify=y, random_state=seed)
    train, val = train_test_split(trval, test_size=val_frac, stratify=y[trval], random_state=seed)
    return np.sort(train), np.sort(val), np.sort(test)


# German Credit

GERMAN_COLS = [
    "checking_status", "duration_months", "credit_history", "purpose", "credit_amount",
    "savings", "employment_since", "installment_rate", "personal_status_sex", "other_debtors",
    "residence_since", "property", "age", "other_installment_plans", "housing",
    "existing_credits", "job", "num_dependents", "telephone", "foreign_worker", "label",
]
GERMAN_NUMERIC = [
    "duration_months", "credit_amount", "installment_rate", "residence_since",
    "age", "existing_credits", "num_dependents",
]
GERMAN_CATEGORICAL = [c for c in GERMAN_COLS[:-1] if c not in GERMAN_NUMERIC]
GERMAN_TARGET = "bad"

# From german.doc: rows are actual class, columns predicted class.
# Classifying a bad customer as good costs 5, a good customer as bad costs 1.
GERMAN_COST_FN = 5.0  # actual bad, predicted good
GERMAN_COST_FP = 1.0  # actual good, predicted bad


def load_german(path=None):
    """Load german.data (the categorical version) with readable column names.

    Adds a binary column 'bad' (1 = bad credit risk) and drops the raw label.
    """
    path = path or DATA_DIR / "german.data"
    df = pd.read_csv(path, sep=" ", header=None, names=GERMAN_COLS)
    df[GERMAN_TARGET] = (df["label"] == 2).astype(int)
    return df.drop(columns="label")


def german_cv_splits(df, n_splits=5, n_repeats=5, seed=0):
    """Repeated stratified k-fold. Yields (repeat, fold, train_idx, test_idx)."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for i, (tr, te) in enumerate(rskf.split(df, df[GERMAN_TARGET])):
        yield i // n_splits, i % n_splits, tr, te
