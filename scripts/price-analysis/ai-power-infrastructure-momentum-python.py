# Full write-up: https://xfinlink.com/blog/ai-power-infrastructure-momentum-python

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "ai-power-infrastructure-momentum-python"
CHART_PATH = Path("worker/src/site/blog-images") / f"{SLUG}.png"
TICKERS = ["CEG", "VST", "NRG", "ETN", "PWR", "GEV", "NEE", "SO", "DUK", "AEP"]
FIELDS = ["market_cap", "revenue_growth", "pe_ratio"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fmt_pct(value: float) -> str:
    return f"{value:+6.1%}"


def fmt_money(value: float) -> str:
    return f"${value / 1000:,.1f}B"


def apply_dark_theme() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#0a0a0a",
            "axes.facecolor": "#0a0a0a",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#e0e0e0",
            "xtick.color": "#e0e0e0",
            "ytick.color": "#e0e0e0",
            "text.color": "#e0e0e0",
            "font.size": 10,
        }
    )


prices = xfl.prices(TICKERS, period="1y", fields=["adj_close", "return_daily"])
require(not prices.empty, "prices returned no rows")

pivot = prices.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
pivot = pivot.dropna(axis=1)
missing_prices = sorted(set(TICKERS) - set(pivot.columns))
require(not missing_prices, f"missing price history for {missing_prices}")
require(len(pivot) > 200, "expected at least 200 daily observations")

returns = pivot.pct_change().dropna()
metrics = xfl.metrics(TICKERS, period_type="ttm", fields=FIELDS)
require(not metrics.empty, "metrics returned no rows")
latest = metrics.sort_values("period_end").groupby("ticker", group_keys=False).tail(1).set_index("ticker")
require(set(TICKERS).issubset(set(latest.index)), "missing latest metrics for one or more tickers")
require(latest[FIELDS].notna().all().all(), "latest metrics contain missing values")
require(((latest["market_cap"] > 1_000) & (latest["market_cap"] < 10_000_000)).all(), "market cap sanity check failed")

rows = []
for ticker in TICKERS:
    series = pivot[ticker]
    drawdown = series / series.cummax() - 1
    rows.append(
        {
            "ticker": ticker,
            "return_1y": series.iloc[-1] / series.iloc[0] - 1,
            "volatility": returns[ticker].std() * np.sqrt(252),
            "max_drawdown": drawdown.min(),
            "revenue_growth": latest.loc[ticker, "revenue_growth"],
            "pe_ratio": latest.loc[ticker, "pe_ratio"],
            "market_cap": latest.loc[ticker, "market_cap"],
        }
    )

screen = pd.DataFrame(rows).sort_values("return_1y", ascending=False)
require(screen[["return_1y", "volatility", "max_drawdown", "revenue_growth", "pe_ratio", "market_cap"]].notna().all().all(), "screen contains missing values")

apply_dark_theme()
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#3b82f6" if value >= 0 else "#ef4444" for value in screen["return_1y"]]
ax.bar(screen["ticker"], screen["return_1y"] * 100, color=colors)
ax.axhline(0, color="#e0e0e0", linewidth=0.8)
ax.set_title("AI Power Infrastructure Momentum")
ax.set_xlabel("Company")
ax.set_ylabel("One-year return (%)")
ax.grid(axis="y", color="#2a2a2a", linewidth=0.6, alpha=0.55)
plt.tight_layout()
CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(CHART_PATH, dpi=150, facecolor=fig.get_facecolor())
plt.close(fig)

print("=== AI Power Infrastructure Momentum Screen ===")
print("Universe: 10 power, grid, and electrification stocks")
print(f"Sample: {pivot.index.min().date()} to {pivot.index.max().date()} ({len(pivot)} trading days)")
print(f"Top 1-year return: {screen.iloc[0]['ticker']} {fmt_pct(screen.iloc[0]['return_1y'])}")
print(f"Median 1-year return: {fmt_pct(screen['return_1y'].median())}")
print(f"Median revenue growth: {fmt_pct(screen['revenue_growth'].median())}")
print()
print("Ticker ranking:")
for _, row in screen.iterrows():
    print(
        f"{row['ticker']:4s} return={fmt_pct(row['return_1y'])}  "
        f"vol={row['volatility']:5.1%}  "
        f"max_drawdown={row['max_drawdown']:6.1%}  "
        f"rev_growth={row['revenue_growth']:5.1%}  "
        f"PE={row['pe_ratio']:6.1f}  "
        f"market_cap={fmt_money(row['market_cap'])}"
    )
