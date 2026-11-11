"""Team-level figures for the report and slides.

fig5_all_frameworks.png : fraud PR-AUC and German Credit cost for all five frameworks
fig6_fairness_age.png   : Lightning approval rates by age group on German Credit

Reads results/summary_all_frameworks.csv (run summarize_results.py first) and
results/person_c/lightning_fairness.csv. Writes to results/figures/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
FIG = RES / "figures"

BLUE, GRAY, INK, INK2, GRID, SURFACE = "#2a78d6", "#b8b7b0", "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 15, "axes.titlesize": 17, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
})


def fig_all_frameworks():
    s = pd.read_csv(RES / "summary_all_frameworks.csv", index_col=0)
    names = list(s.index)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    vals, errs = s["fraud_pr_auc_mean"], s["fraud_pr_auc_std"]
    best = vals.idxmax()
    ax.bar(names, vals, yerr=errs, capsize=4, width=0.6,
           color=[BLUE if n == best else GRAY for n in names],
           error_kw={"ecolor": INK2, "linewidth": 1.2})
    for i, v in enumerate(vals):
        ax.text(i, v + errs.iloc[i] + 0.01, f"{v:.3f}", ha="center", fontsize=13)
    ax.set_ylim(0.6, 0.95)
    ax.set_ylabel("Fraud PR-AUC (mean ± std, 10 seeds)")
    ax.set_title("Fraud: XGBoost ranks frauds best")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    vals, errs = s["german_cost_1_6_mean"], s["german_cost_1_6_std"]
    best = vals.idxmin()
    ax.bar(names, vals, yerr=errs, capsize=4, width=0.6,
           color=[BLUE if n == best else GRAY for n in names],
           error_kw={"ecolor": INK2, "linewidth": 1.2})
    for i, v in enumerate(vals):
        ax.text(i, v / 2, f"{v:.3f}", ha="center", va="center", fontsize=13,
                color=SURFACE if names[i] == best else INK, fontweight="bold")
    ax.axhline(0.70, color=INK, linestyle="--", linewidth=1.5)
    ax.text(-0.45, 0.83, "reject everyone = 0.70", ha="left", va="bottom", fontsize=12)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("Cost per applicant at cut-off 1/6 (25 folds)")
    ax.set_title("German Credit: Keras is cheapest")
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    fig.savefig(FIG / "fig5_all_frameworks.png", dpi=200)
    plt.close(fig)


def fig_fairness_age():
    f = pd.read_csv(RES / "person_c" / "lightning_fairness.csv")
    a = f[f.attribute == "age_group"].groupby("group")[["approval_rate", "bad_missed_rate"]].mean()
    a = a.loc[["18-25", "26-35", "36-50", "51+"]] * 100
    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = range(len(a))
    w = 0.38
    ax.bar([i - w / 2 for i in x], a.approval_rate, w, color=BLUE, label="All applicants approved")
    ax.bar([i + w / 2 for i in x], a.bad_missed_rate, w, color=GRAY, label="Bad borrowers approved")
    for i, (p, q) in enumerate(zip(a.approval_rate, a.bad_missed_rate)):
        ax.text(i - w / 2, p + 1, f"{p:.0f}%", ha="center", fontsize=13)
        ax.text(i + w / 2, q + 1, f"{q:.0f}%", ha="center", fontsize=13)
    ax.set_xticks(list(x), [g.replace("-", " to ") for g in a.index])
    ax.set_xlabel("Age group")
    ax.set_ylabel("Share approved (%)")
    ax.set_ylim(0, 80)
    ax.set_title("Lightning on German Credit, cut-off 1/6")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_fairness_age.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    fig_all_frameworks()
    fig_fairness_age()
    print("figures written to", FIG)
