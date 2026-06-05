# Full write-up: https://xfinlink.com/blog/rate-sensitive-sector-rotation-python

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "rate-sensitive-sector-rotation-python"
SECTORS = ["XLK", "XLF", "XLI", "XLV", "XLP", "XLU", "XLE"]
TICKERS = ["TLT", "SPY"] + SECTORS
WINDOW = 21


def fmt_pct(value: float) -> str:
    return f"{value * 100:6.1f}%"


def make_chart(result: pd.DataFrame) -> None:
    plt.rcParams.update({
        "figure.facecolor": "#0a0a0a",
        "axes.facecolor": "#0a0a0a",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#e0e0e0",
        "ytick.color": "#e0e0e0",
        "text.color": "#e0e0e0",
        "axes.titleweight": "bold",
        "font.size": 10,
    })

    fig, ax = plt.subplots(figsize=(10, 5))
    ordered = result.sort_values("avg_return_when_tlt_rallies", ascending=False)
    colors = ["#3b82f6" if value > 0 else "#6b7280" for value in ordered["avg_return_when_tlt_rallies"]]
    bars = ax.bar(ordered.index, ordered["avg_return_when_tlt_rallies"] * 100, color=colors)
    ax.axhline(0, color="#e0e0e0", linewidth=1, alpha=0.6)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, color="#e0e0e0", fontsize=8)
    ax.set_title("Average sector return during bond-rally windows")
    ax.set_ylabel("21-day return")
    ax.set_xlabel("Sector ETF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    out = Path("worker/src/site/blog-images") / f"{SLUG}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, facecolor="#0a0a0a")
    plt.close(fig)


prices = xfl.prices(TICKERS, period="3y", fields=["adj_close"])
if prices.empty:
    raise ValueError("Price DataFrame is empty")

pivot = prices.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
missing = sorted(set(TICKERS) - set(pivot.columns))
if missing:
    raise ValueError(f"Missing price columns: {missing}")

pivot = pivot[TICKERS].dropna()
if pivot.empty:
    raise ValueError("No overlapping sector and Treasury-bond history")
if not pivot.index.is_monotonic_increasing:
    raise ValueError("Dates are not ordered correctly")

daily = pivot.pct_change(fill_method=None).dropna()
window_returns = pivot.pct_change(WINDOW, fill_method=None).dropna()
threshold = window_returns["TLT"].quantile(0.80)
bond_rallies = window_returns["TLT"] >= threshold

rows = []
for ticker in SECTORS:
    rows.append({
        "ticker": ticker,
        "avg_return_when_tlt_rallies": window_returns.loc[bond_rallies, ticker].mean(),
        "avg_return_all_windows": window_returns[ticker].mean(),
        "latest_63d_momentum": pivot[ticker].iloc[-1] / pivot[ticker].iloc[-64] - 1,
        "daily_beta_to_tlt": daily[ticker].cov(daily["TLT"]) / daily["TLT"].var(),
    })

result = pd.DataFrame(rows).set_index("ticker")
if result.isna().any().any():
    raise ValueError("Sector rotation result contains NaN values")

make_chart(result)

ranked = result.sort_values("avg_return_when_tlt_rallies", ascending=False)
best = ranked.index[0]
worst = ranked.index[-1]

print("=== Rate-Sensitive Sector Rotation Screen ===")
print(f"Sample: {daily.index.min().date()} to {daily.index.max().date()} ({len(daily)} trading days)")
print(f"Bond-rally definition: top 20% of {WINDOW}-day TLT returns")
print(f"TLT rally threshold:              {fmt_pct(threshold)}")
print(f"Number of bond-rally windows:     {int(bond_rallies.sum())}")
print(f"Best sector in bond rallies:      {best} ({fmt_pct(ranked.loc[best, 'avg_return_when_tlt_rallies'])})")
print(f"Worst sector in bond rallies:     {worst} ({fmt_pct(ranked.loc[worst, 'avg_return_when_tlt_rallies'])})")
print()
print("Sector ranking during bond-rally windows:")
for ticker, row in ranked.iterrows():
    print(
        f"{ticker:<4} rally_return={fmt_pct(row['avg_return_when_tlt_rallies'])}  "
        f"all_windows={fmt_pct(row['avg_return_all_windows'])}  "
        f"latest_63d_momentum={fmt_pct(row['latest_63d_momentum'])}  "
        f"beta_to_TLT={row['daily_beta_to_tlt']:5.2f}"
    )
