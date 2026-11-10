"""Figures for the report and slides. Reads the CSVs in results/person_a.

Writes PNGs to results/person_a/figures.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "person_a"
FIG = RES / "figures"

COLORS = {"xgboost": "#2a78d6", "fastai": "#eb6834"}
LABELS = {"xgboost": "XGBoost", "fastai": "fastai"}
MARKERS = {"xgboost": "o", "fastai": "s"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 13, "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False,
})


def fig_split_variance():
    r = pd.read_csv(RES / "fraud_runs.csv")
    r = r[r.features == "base"]
    fig, ax = plt.subplots(figsize=(9, 5))
    groups = [("temporal", "xgboost"), ("temporal", "fastai"), ("random", "xgboost"), ("random", "fastai")]
    rng = np.random.default_rng(0)
    for i, (split, model) in enumerate(groups):
        v = r[(r.split == split) & (r.model == model)].pr_auc.to_numpy()
        x = i + rng.uniform(-0.12, 0.12, len(v))
        ax.scatter(x, v, s=70, color=COLORS[model], marker=MARKERS[model],
                   edgecolor=SURFACE, linewidth=2, zorder=3,
                   label=LABELS[model] if split == "temporal" else None)
        ax.hlines(v.mean(), i - 0.25, i + 0.25, color=INK, linewidth=2, zorder=4)
        ax.text(i + 0.28, v.mean(), f"{v.mean():.3f}", va="center", color=INK2, fontsize=11)
    ax.set_xticks(range(4), ["Temporal\nXGBoost", "Temporal\nfastai", "Random\nXGBoost", "Random\nfastai"])
    ax.set_ylabel("Test PR-AUC")
    ax.set_title("Random splits spread PR-AUC far more than model seeds do")
    ax.legend(loc="lower left")
    ax.grid(axis="x", visible=False)
    fig.text(0.01, 0.01, "10 seeds per group, base features. Black bar = mean. "
             "Temporal: fixed split, seed changes the model only.", color=INK2, fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "fig1_split_variance.png", dpi=200)
    plt.close(fig)


def fig_fraud_cost():
    s = pd.read_csv(RES / "cost_summary.csv")
    s = s[s.features == "base"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, ct, title in ((axes[0], False, "Cost = review fees + missed amounts"),
                          (axes[1], True, "Plus follow-on loss for missed probes")):
        g = s[s.card_testing == ct]
        for model in ("xgboost", "fastai"):
            h = g[g.model == model].sort_values("review_fee")
            ax.errorbar(h.review_fee, h.cost_mean, yerr=h.cost_std, color=COLORS[model],
                        marker=MARKERS[model], markersize=9, linewidth=2, capsize=4,
                        markeredgecolor=SURFACE, markeredgewidth=2, label=LABELS[model])
        nothing = g.cost_flag_nothing.iloc[0]
        ax.set_title(title)
        ax.set_xlabel("Review fee per alert (R)")
        ax.set_xticks([1, 2, 5, 10])
        ax.text(0.02, 0.97, f"Flagging nothing costs {nothing:,.0f}", transform=ax.transAxes,
                va="top", color=INK2, fontsize=11)
    axes[0].set_ylabel("Total test-period cost (mean ± std, 10 seeds)")
    axes[0].legend(loc="lower right")
    fig.suptitle("Which model is cheaper depends on the cost assumption", fontweight="bold", fontsize=16)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_fraud_cost.png", dpi=200)
    plt.close(fig)


def fig_german_threshold():
    r = pd.read_csv(RES / "german_runs.csv")
    policies = [("default", "0.5\n(default)"), ("theory", "0.167\n(cost matrix)"), ("tuned", "Tuned on\nvalidation")]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    w = 0.36
    for j, model in enumerate(("xgboost", "fastai")):
        means = [r[r.model == model][f"cost_{p}"].mean() for p, _ in policies]
        stds = [r[r.model == model][f"cost_{p}"].std() for p, _ in policies]
        x = np.arange(3) + (j - 0.5) * (w + 0.02)
        ax.bar(x, means, w, yerr=stds, color=COLORS[model], label=LABELS[model],
               capsize=4, error_kw={"ecolor": INK2, "linewidth": 1.2})
        for xi, m in zip(x, means):
            ax.text(xi, 0.02, f"{m:.2f}", ha="center", va="bottom", color=SURFACE,
                    fontsize=12, fontweight="bold")
    reject = r.cost_reject_all.mean()
    ax.axhline(reject, color=INK, linewidth=2, linestyle="--")
    ax.text(2.62, reject + 0.02, f"Reject everyone = {reject:.2f}", ha="right", va="bottom", color=INK, fontsize=11)
    ax.set_xticks(range(3), [lab for _, lab in policies])
    ax.set_ylabel("Cost per applicant (mean ± std, 25 folds)")
    ax.set_title("At the default threshold both models lose to rejecting everyone")
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Decision threshold on P(bad)")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_german_threshold.png", dpi=200)
    plt.close(fig)


def fig_attack_control():
    a = pd.read_csv(RES / "attack_control.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    cats = [("with_siblings", "recall_attack", "Attack frauds,\nsiblings in train"),
            ("without_siblings", "recall_attack", "Attack frauds,\nsiblings removed"),
            ("with_siblings", "recall_other", "All other\nfrauds")]
    w = 0.36
    for j, model in enumerate(("xgboost", "fastai")):
        sub = a[a.model == model]
        vals = [sub[sub.variant == v][c].mean() for v, c, _ in cats]
        sds = [sub[sub.variant == v][c].std() for v, c, _ in cats]
        x = np.arange(3) + (j - 0.5) * (w + 0.02)
        ax.bar(x, vals, w, yerr=sds, color=COLORS[model], label=LABELS[model], capsize=4,
               error_kw={"ecolor": INK2, "linewidth": 1.2})
        for xi, v in zip(x, vals):
            ax.text(xi, 0.03, f"{v:.2f}", ha="center", va="bottom", color=SURFACE, fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1.2)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_xticks(range(3), [lab for _, _, lab in cats])
    ax.set_ylabel("Recall at 0.1% false-positive rate")
    ax.set_title("Removing the hour-11 attack from training does not hurt its recall")
    ax.legend(loc="upper center", ncol=2)
    ax.grid(axis="x", visible=False)
    fig.text(0.01, 0.01, "10 random splits. The attack frauds are easy on their own; "
             "the model is not memorising them.", color=INK2, fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "fig4_attack_control.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_split_variance()
    fig_fraud_cost()
    fig_german_threshold()
    fig_attack_control()
    print("figures written to", FIG)
