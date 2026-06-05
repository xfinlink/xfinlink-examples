# Full write-up: https://xfinlink.com/blog/sp500-low-volatility-drawdown-python

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "sp500-low-volatility-drawdown-python"
AS_OF = "2025-06-05"
CANDIDATES = [
    "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOG", "AVGO", "JPM", "V", "XOM",
    "UNH", "MA", "HD", "PG", "COST", "JNJ", "ABBV", "BAC", "KO", "CRM",
    "NFLX", "WMT", "ORCL", "CSCO", "MRK", "CVX", "AMD", "PEP", "MCD", "TMO",
    "ABT", "ACN", "LIN", "DIS", "ADBE", "WFC", "CAT", "QCOM", "GE", "TXN",
    "AMGN", "PM", "INTU", "VZ", "IBM", "NOW", "ISRG", "GS", "SPGI", "HON",
    "RTX", "LOW", "BKNG", "AXP", "PFE", "NEE", "UNP", "T", "BLK", "LMT",
]


def max_drawdown(series: pd.Series) -> float:
    wealth = series / series.iloc[0]
    return (wealth / wealth.cummax() - 1).min()


def fmt_pct(value: float) -> str:
    return f"{value * 100:6.1f}%"


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
    x = range(len(summary.index))
    ax.bar([i - 0.18 for i in x], summary["total_return"] * 100, width=0.36, color="#3b82f6", label="Total return")
    ax.bar([i + 0.18 for i in x], summary["max_drawdown"] * 100, width=0.36, color="#6b7280", label="Max drawdown")
    ax.axhline(0, color="#e0e0e0", linewidth=1, alpha=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary.index)
    ax.set_title("Low-volatility versus high-volatility S&P 500 basket")
    ax.set_ylabel("Percent")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    out = Path("worker/src/site/blog-images") / f"{SLUG}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, facecolor="#0a0a0a")
    plt.close(fig)


constituents = xfl.index("sp500", as_of=AS_OF, limit=500)
if constituents.empty:
    raise ValueError("Index constituents DataFrame is empty")

member_set = set(constituents["ticker"].dropna())
tickers = [ticker for ticker in CANDIDATES if ticker in member_set]
if len(tickers) < 40:
    raise ValueError("Too few candidate tickers were S&P 500 members as of the test date")

prices = xfl.prices(tickers, start=AS_OF, fields=["adj_close"])
if prices.empty:
    raise ValueError("Price DataFrame is empty")

pivot = prices.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
available = [ticker for ticker in tickers if ticker in pivot.columns]
if len(available) < 40:
    raise ValueError("Too few requested common-stock tickers returned price data")
pivot = pivot[available]
if not pivot.index.is_monotonic_increasing:
    raise ValueError("Dates are not ordered correctly")

coverage = pivot.notna().sum()
latest_dates = pivot.apply(lambda series: series.dropna().index.max())
global_latest = pivot.index.max()
valid = [
    ticker for ticker in available
    if coverage[ticker] >= 200 and latest_dates[ticker] >= global_latest - pd.Timedelta(days=7)
]
pivot = pivot[valid].dropna()
if len(valid) < 40 or pivot.empty:
    raise ValueError("Not enough valid price histories for the point-in-time universe")

returns = pivot.pct_change(fill_method=None).dropna()
if returns.empty or returns.isna().any().any():
    raise ValueError("Return matrix is empty or contains NaN values")

rows = []
for ticker in valid:
    rows.append({
        "ticker": ticker,
        "annualized_volatility": returns[ticker].std() * (252 ** 0.5),
        "total_return": pivot[ticker].iloc[-1] / pivot[ticker].iloc[0] - 1,
        "max_drawdown": max_drawdown(pivot[ticker]),
    })

stats = pd.DataFrame(rows).set_index("ticker")
low_vol = stats.nsmallest(10, "annualized_volatility")
high_vol = stats.nlargest(10, "annualized_volatility")

summary = pd.DataFrame({
    "annualized_volatility": [low_vol["annualized_volatility"].mean(), high_vol["annualized_volatility"].mean()],
    "total_return": [low_vol["total_return"].mean(), high_vol["total_return"].mean()],
    "max_drawdown": [low_vol["max_drawdown"].mean(), high_vol["max_drawdown"].mean()],
}, index=["Low-vol basket", "High-vol basket"])

make_chart(summary)

print("=== Point-in-Time S&P 500 Low-Volatility Drawdown Test ===")
print(f"Constituent date: {AS_OF}")
print(f"Price sample: {returns.index.min().date()} to {returns.index.max().date()} ({len(returns)} trading days)")
print(f"Tested liquid member subset: {len(valid)} stocks")
print()
print("Basket comparison:")
for label, row in summary.iterrows():
    print(
        f"{label:<16} volatility={fmt_pct(row['annualized_volatility'])}  "
        f"total_return={fmt_pct(row['total_return'])}  "
        f"max_drawdown={fmt_pct(row['max_drawdown'])}"
    )
print()
print("Lowest-volatility stocks:")
for ticker, row in low_vol.sort_values("annualized_volatility").iterrows():
    print(
        f"{ticker:<5} vol={fmt_pct(row['annualized_volatility'])}  "
        f"return={fmt_pct(row['total_return'])}  "
        f"max_drawdown={fmt_pct(row['max_drawdown'])}"
    )
