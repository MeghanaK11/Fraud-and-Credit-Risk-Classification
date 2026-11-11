"""EDA figures for the slides: card-testing amounts and fraud rate by hour.

Uses the raw file (before dedupe) because these describe the dataset as given.
Writes results/person_a/figures/eda_*.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "results" / "person_a" / "figures"
BLUE, GRAY, INK, INK2, GRID, SURFACE = "#2a78d6", "#b8b7b0", "#0b0b0b", "#52514e", "#e4e3de", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "font.size": 15, "axes.titlesize": 17, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True,
})


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(ROOT / "data" / "creditcard_sample.csv")

    # 1. Share of transactions at 1.00 or less
    share = d.assign(micro=d.Amount <= 1).groupby("Class").micro.mean()
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(["Normal", "Fraud"], share.values * 100, color=[GRAY, BLUE], width=0.55)
    for b, v in zip(bars, share.values * 100):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}%", ha="center", fontsize=18, fontweight="bold")
    ax.set_ylabel("Transactions of 1.00 or less (%)")
    ax.set_ylim(0, 50)
    ax.set_title("Tiny charges are 4x more common in fraud")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "eda_micro_amounts.png", dpi=200)
    plt.close(fig)

    # 2. Fraud rate by hour
    d["hour"] = (d.Time // 3600).astype(int)
    h = d.groupby("hour").Class.agg(["mean", "sum"])
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [BLUE if hr == 11 or 1 <= hr <= 6 else GRAY for hr in h.index]
    ax.bar(h.index, h["mean"] * 100, color=colors, width=0.8)
    overall = d.Class.mean() * 100
    ax.axhline(overall, color=INK, linestyle="--", linewidth=1.5)
    ax.text(h.index.max() + 0.4, overall + 0.03, f"overall {overall:.2f}%", ha="right", va="bottom", fontsize=13)
    ax.annotate("hour 11: 43 frauds,\nmedian 25 s apart", xy=(11, h.loc[11, "mean"] * 100),
                xytext=(12.5, 1.1), fontsize=13, color=INK, arrowprops=dict(arrowstyle="-", color=INK2))
    ax.set_xlabel("Hour since the first transaction")
    ax.set_ylabel("Fraud rate (%)")
    ax.set_xticks(h.index)
    ax.set_title("Fraud is higher at night and comes in bursts")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(FIG / "eda_fraud_by_hour.png", dpi=200)
    plt.close(fig)
    print("written", FIG)


if __name__ == "__main__":
    main()
