"""Shared evaluation: classification metrics, probability metrics, costs, CIs."""
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
    roc_curve,
)


def recall_at_fpr(y, p, max_fpr):
    """Highest recall reachable while the false-positive rate stays <= max_fpr."""
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= max_fpr
    return float(tpr[ok].max()) if ok.any() else 0.0


def probability_metrics(y, p):
    """The brief's 'regression' metrics, computed on predicted probabilities.

    RMSE here is sqrt(Brier score). R^2 compares the model with always
    predicting the base rate, so it can be negative.
    """
    return {
        "brier": brier_score_loss(y, p),
        "rmse": float(np.sqrt(mean_squared_error(y, p))),
        "mae": mean_absolute_error(y, p),
        "r2": r2_score(y, p),
    }


def all_metrics(y, p, fpr_targets=(0.001, 0.01)):
    out = {
        "pr_auc": average_precision_score(y, p),
        "roc_auc": roc_auc_score(y, p),
    }
    for f in fpr_targets:
        out[f"recall_at_fpr_{f:g}"] = recall_at_fpr(y, p, f)
    out.update(probability_metrics(y, p))
    return out


def bootstrap_ci(y, p, metric_fn, n_boot=1000, seed=0, alpha=0.05):
    """Percentile bootstrap CI over test rows. Resamples that contain no
    positive case are skipped because PR-AUC is undefined there."""
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    n = len(y)
    vals = []
    while len(vals) < n_boot:
        i = rng.integers(0, n, n)
        if y[i].sum() == 0:
            continue
        vals.append(metric_fn(y[i], p[i]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


# Cost functions

def fraud_cost(y, flagged, amount, review_cost, testing_loss=None, testing_cap=1.0):
    """Total cost of a set of fraud alerts.

    Each alert costs review_cost (an analyst looks at it). Each missed fraud
    costs its own amount. If testing_loss is given, a caught fraud with
    amount <= testing_cap is treated as a stopped card-testing probe: it
    prevents a follow-on fraud worth testing_loss, so it earns that back.
    Equivalently, missing such a probe costs its amount plus testing_loss.
    """
    y = np.asarray(y).astype(bool)
    flagged = np.asarray(flagged).astype(bool)
    amount = np.asarray(amount, dtype=float)
    missed = y & ~flagged
    loss = flagged.sum() * review_cost + amount[missed].sum()
    if testing_loss is not None:
        probe = amount <= testing_cap
        loss += testing_loss * (missed & probe).sum()
    return float(loss)


def best_threshold(y, p, cost_fn, grid=None):
    """Threshold on p that minimises cost_fn(y, p >= t). Returns (t, cost)."""
    if grid is None:
        grid = np.unique(np.concatenate([np.linspace(0, 1, 201), np.quantile(p, np.linspace(0.9, 1, 201))]))
    costs = [cost_fn(y, p >= t) for t in grid]
    k = int(np.argmin(costs))
    return float(grid[k]), float(costs[k])


def german_cost(y, flagged_bad, cost_fn=5.0, cost_fp=1.0):
    """Average cost per applicant using the German Credit cost matrix."""
    y = np.asarray(y).astype(bool)
    f = np.asarray(flagged_bad).astype(bool)
    return float((cost_fn * (y & ~f).sum() + cost_fp * (~y & f).sum()) / len(y))
