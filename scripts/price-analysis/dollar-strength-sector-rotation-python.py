# Full write-up: https://xfinlink.com/blog/dollar-strength-sector-rotation-python

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "dollar-strength-sector-rotation-python"
DOLLAR_PROXY = "UUP"
SECTORS = ["XLK", "XLF", "XLV", "XLE", "XLY", "XLP", "XLI", "XLU"]
TICKERS = [DOLLAR_PROXY, *SECTORS]


def fmt_pct(value: float) -> str:
    return f"{value * 100:6.2f}%"


def make_chart(summary: pd.DataFrame) -> None:
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
    ordered = summary.sort_values("rally_minus_weak")
    colors = ["#3b82f6" if value >= 0 else "#6b7280" for value in ordered["rally_minus_weak"]]
    ax.bar(ordered["sector"], ordered["rally_minus_weak"] * 100, color=colors)
    ax.axhline(0, color="#e0e0e0", linewidth=1, alpha=0.45)
    ax.set_title("Forward sector returns after dollar rally weeks")
    ax.set_xlabel("Sector ETF")
    ax.set_ylabel("Top-minus-bottom dollar regime return")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    out = Path("worker/src/site/blog-images") / f"{SLUG}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, facecolor="#0a0a0a")
    plt.close(fig)


prices = xfl.prices(TICKERS, period="5y", interval="1w", fields=["adj_close"], max_rows=50000)
if prices.empty:
    raise ValueError("Price DataFrame is empty")

required_price = {"ticker", "date", "adj_close"}
missing_price = required_price - set(prices.columns)
if missing_price:
    raise ValueError(f"Missing price columns: {sorted(missing_price)}")

price_table = prices.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
missing_tickers = sorted(set(TICKERS) - set(price_table.columns))
if missing_tickers:
    raise ValueError(f"Missing price data for: {missing_tickers}")
if not price_table.index.is_monotonic_increasing:
    raise ValueError("Price dates are not ordered correctly")
if (price_table <= 0).any().any():
    raise ValueError("Adjusted prices should be positive")

price_table = price_table.dropna()
if len(price_table) < 200:
    raise ValueError(f"Not enough weekly price history: {len(price_table)} rows")

weekly_returns = price_table.pct_change().dropna()
dollar_returns = weekly_returns[DOLLAR_PROXY]
sector_forward_returns = weekly_returns[SECTORS].shift(-1)
analysis = pd.concat([dollar_returns.rename("dollar_return"), sector_forward_returns], axis=1).dropna()

top_cutoff = analysis["dollar_return"].quantile(0.80)
bottom_cutoff = analysis["dollar_return"].quantile(0.20)
dollar_rally = analysis[analysis["dollar_return"] >= top_cutoff]
dollar_weak = analysis[analysis["dollar_return"] <= bottom_cutoff]

if dollar_rally.empty or dollar_weak.empty:
    raise ValueError("Need both dollar rally and weak-dollar regimes")

rows = []
for sector in SECTORS:
    rally_mean = dollar_rally[sector].mean()
    weak_mean = dollar_weak[sector].mean()
    rows.append({
        "sector": sector,
        "after_dollar_rally": rally_mean,
        "after_weak_dollar": weak_mean,
        "rally_minus_weak": rally_mean - weak_mean,
    })

summary = pd.DataFrame(rows).sort_values("rally_minus_weak")
if summary.isna().any().any():
    raise ValueError("Sector rotation summary contains NaN values")

make_chart(summary)

print("=== Dollar Strength Sector Rotation Test ===")
print(f"Weekly sample: {analysis.index.min().date()} to {analysis.index.max().date()} ({len(analysis)} weeks)")
print(f"Dollar proxy: {DOLLAR_PROXY}")
print(f"Dollar rally threshold: weekly return >= {fmt_pct(top_cutoff)}")
print(f"Weak-dollar threshold: weekly return <= {fmt_pct(bottom_cutoff)}")
print(f"Dollar rally weeks: {len(dollar_rally)}; weak-dollar weeks: {len(dollar_weak)}")
print()
print("Average next-week sector returns:")
for _, row in summary.iterrows():
    print(
        f"{row['sector']:<4} after_rally={fmt_pct(row['after_dollar_rally'])}  "
        f"after_weak={fmt_pct(row['after_weak_dollar'])}  "
        f"spread={fmt_pct(row['rally_minus_weak'])}"
    )
print()
best = summary.iloc[-1]
worst = summary.iloc[0]
print(f"Best relative sector after dollar rallies: {best['sector']} ({fmt_pct(best['rally_minus_weak'])} spread)")
print(f"Weakest relative sector after dollar rallies: {worst['sector']} ({fmt_pct(worst['rally_minus_weak'])} spread)")
