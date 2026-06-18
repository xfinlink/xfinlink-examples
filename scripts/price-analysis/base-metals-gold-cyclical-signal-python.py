# Full write-up: https://xfinlink.com/blog/base-metals-gold-cyclical-signal-python

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "base-metals-gold-cyclical-signal-python"
CHART_PATH = Path("worker/src/site/blog-images") / f"{SLUG}.png"
TICKERS = ["DBB", "GLD", "XLI", "XLB", "SPY"]
ASSETS = ["XLI", "XLB", "SPY"]
SIGNAL_MONTHS = 6
FORWARD_MONTHS = 3


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fmt_pct(value: float) -> str:
    return f"{value:+6.1%}"


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


prices = xfl.prices(TICKERS, period="10y", fields=["adj_close", "return_daily"])
require(not prices.empty, "prices returned no rows")
price_daily = prices.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
return_daily = prices.pivot_table(index="date", columns="ticker", values="return_daily").sort_index()
missing = sorted(set(TICKERS) - set(price_daily.columns))
require(not missing, f"missing price history for {missing}")
price_daily = price_daily[TICKERS].dropna()
return_daily = return_daily[TICKERS].dropna()
require(len(price_daily) > 2000, "expected at least 2000 complete daily observations")

monthly_prices = price_daily.resample("ME").last()
monthly_returns = (1 + return_daily).resample("ME").prod() - 1
if monthly_prices.index[-1] > price_daily.index.max():
    monthly_prices = monthly_prices.iloc[:-1]
    monthly_returns = monthly_returns.iloc[:-1]

ratio = monthly_prices["DBB"] / monthly_prices["GLD"]
signal = ratio.pct_change(SIGNAL_MONTHS)
forward = pd.DataFrame(index=monthly_returns.index)
for asset in ASSETS:
    forward[asset] = (1 + monthly_returns[asset]).rolling(FORWARD_MONTHS).apply(np.prod, raw=True).shift(-FORWARD_MONTHS) - 1

analysis = pd.concat({"signal": signal, **{asset: forward[asset] for asset in ASSETS}}, axis=1).dropna()
require(len(analysis) >= 80, "expected at least 80 monthly signal observations")
labels = ["Q1 weakest", "Q2", "Q3", "Q4", "Q5 strongest"]
analysis["quintile"], bins = pd.qcut(analysis["signal"], 5, labels=labels, retbins=True)
grouped = analysis.groupby("quintile", observed=False)[ASSETS].mean()
counts = analysis.groupby("quintile", observed=False).size()
latest_signal = signal.dropna().iloc[-1]
latest_bins = bins.copy()
latest_bins[0] = -np.inf
latest_bins[-1] = np.inf
latest_quintile = pd.cut([latest_signal], bins=latest_bins, labels=labels, include_lowest=True)[0]

require(grouped.notna().all().all(), "quintile output contains missing values")
require(counts.min() >= 10, "each quintile should have at least 10 observations")

apply_dark_theme()
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(grouped.index))
width = 0.24
colors = {"XLI": "#3b82f6", "XLB": "#22c55e", "SPY": "#f59e0b"}
for i, asset in enumerate(ASSETS):
    ax.bar(x + (i - 1) * width, grouped[asset], width=width, label=asset, color=colors[asset])
ax.axhline(0, color="#777777", linewidth=0.8)
ax.set_title("Base Metals / Gold Signal and Forward Returns")
ax.set_xlabel("Six-Month DBB/GLD Signal Quintile")
ax.set_ylabel("Average Next 3-Month Return")
ax.set_xticks(x)
ax.set_xticklabels(grouped.index)
ax.legend(frameon=False)
ax.grid(axis="y", color="#2a2a2a", linewidth=0.6, alpha=0.55)
plt.tight_layout()
CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(CHART_PATH, dpi=150, facecolor=fig.get_facecolor())
plt.close(fig)

print("=== Base-Metals-to-Gold Cyclical Signal ===")
print("Signal: 6-month change in DBB/GLD")
print("Forward return window: next 3 months")
print(f"Sample: {analysis.index.min().date()} to {analysis.index.max().date()} ({len(analysis)} monthly observations)")
print(f"Latest signal: {fmt_pct(latest_signal)} ({latest_quintile})")
print()
print("Average next 3-month returns by signal quintile:")
for quintile, row in grouped.iterrows():
    print(
        f"{quintile:12s}  n={counts.loc[quintile]:2d}  "
        f"XLI={fmt_pct(row['XLI'])}  "
        f"XLB={fmt_pct(row['XLB'])}  "
        f"SPY={fmt_pct(row['SPY'])}"
    )
print()
spread = grouped.loc["Q5 strongest"] - grouped.loc["Q1 weakest"]
print("Strong-minus-weak signal spread:")
for asset, value in spread.items():
    print(f"{asset:3s} spread={fmt_pct(value)}")
