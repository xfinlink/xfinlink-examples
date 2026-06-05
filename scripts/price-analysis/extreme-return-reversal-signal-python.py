# Full write-up: https://xfinlink.com/blog/extreme-return-reversal-signal-python

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "extreme-return-reversal-signal-python"
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "JPM", "XOM", "JNJ", "PG", "SPY"]
LOOKBACK = 21
FORWARD = 21


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
    ordered = result.sort_values("reversal_spread", ascending=False)
    colors = ["#3b82f6" if value > 0 else "#6b7280" for value in ordered["reversal_spread"]]
    bars = ax.bar(ordered.index, ordered["reversal_spread"] * 100, color=colors)
    ax.axhline(0, color="#e0e0e0", linewidth=1, alpha=0.6)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, color="#e0e0e0", fontsize=8)
    ax.set_title("Forward return after one-month price extremes")
    ax.set_ylabel("Loser forward return minus winner forward return")
    ax.set_xlabel("Ticker")
    ax.tick_params(axis="x", rotation=35)
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
    raise ValueError("No overlapping price history")
if not pivot.index.is_monotonic_increasing:
    raise ValueError("Dates are not ordered correctly")

lookback = pivot.pct_change(LOOKBACK, fill_method=None)
forward = pivot.shift(-FORWARD) / pivot - 1

rows = []
for ticker in TICKERS:
    sample = pd.DataFrame({
        "lookback": lookback[ticker],
        "forward": forward[ticker],
    }).dropna()
    if len(sample) < 250:
        raise ValueError(f"Not enough observations for {ticker}")

    lower = sample["lookback"].quantile(0.10)
    upper = sample["lookback"].quantile(0.90)
    losers = sample[sample["lookback"] <= lower]
    winners = sample[sample["lookback"] >= upper]

    rows.append({
        "ticker": ticker,
        "loser_threshold": lower,
        "winner_threshold": upper,
        "forward_after_losers": losers["forward"].mean(),
        "forward_after_winners": winners["forward"].mean(),
        "reversal_spread": losers["forward"].mean() - winners["forward"].mean(),
        "loser_events": len(losers),
        "winner_events": len(winners),
    })

result = pd.DataFrame(rows).set_index("ticker")
if result.isna().any().any():
    raise ValueError("Signal result contains NaN values")

make_chart(result)

aggregate_loser = result["forward_after_losers"].mean()
aggregate_winner = result["forward_after_winners"].mean()
aggregate_spread = result["reversal_spread"].mean()
best = result["reversal_spread"].idxmax()
worst = result["reversal_spread"].idxmin()

print("=== One-Month Extreme Return Reversal Test ===")
print(f"Sample: {pivot.index.min().date()} to {pivot.index.max().date()} ({len(pivot)} price observations)")
print(f"Signal definition: bottom and top 10% of prior {LOOKBACK}-day returns")
print(f"Forward horizon: {FORWARD} trading days")
print(f"Average forward return after loser extremes:  {fmt_pct(aggregate_loser)}")
print(f"Average forward return after winner extremes: {fmt_pct(aggregate_winner)}")
print(f"Average reversal spread:                     {fmt_pct(aggregate_spread)}")
print(f"Strongest reversal ticker:                   {best} ({fmt_pct(result.loc[best, 'reversal_spread'])})")
print(f"Weakest reversal ticker:                     {worst} ({fmt_pct(result.loc[worst, 'reversal_spread'])})")
print()
print("Ticker-level signal results:")
for ticker, row in result.sort_values("reversal_spread", ascending=False).iterrows():
    print(
        f"{ticker:<5} loser_fwd={fmt_pct(row['forward_after_losers'])}  "
        f"winner_fwd={fmt_pct(row['forward_after_winners'])}  "
        f"spread={fmt_pct(row['reversal_spread'])}  "
        f"events={int(row['loser_events'])}/{int(row['winner_events'])}"
    )
