# Full write-up: https://xfinlink.com/blog/commodity-tail-risk-expected-shortfall-python

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "commodity-tail-risk-expected-shortfall-python"
CHART_PATH = Path("worker/src/site/blog-images") / f"{SLUG}.png"
TICKERS = ["GLD", "SLV", "USO", "UNG", "DBA", "DBB", "DBC"]
TAIL_PROBABILITY = 0.05


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fmt_pct(value: float) -> str:
    return f"{value:+6.1%}"


def fmt_abs_pct(value: float) -> str:
    return f"{value:5.1%}"


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


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


prices = xfl.prices(TICKERS, period="5y", fields=["return_daily"])
require(not prices.empty, "prices returned no rows")
returns = prices.pivot_table(index="date", columns="ticker", values="return_daily").sort_index()
missing = sorted(set(TICKERS) - set(returns.columns))
require(not missing, f"missing return history for {missing}")
returns = returns[TICKERS].dropna()
require(len(returns) > 1000, "expected at least 1000 complete daily observations")
require(returns.index.is_monotonic_increasing, "dates are not sorted")

rows = []
for ticker in TICKERS:
    r = returns[ticker]
    cutoff = r.quantile(TAIL_PROBABILITY)
    tail = r[r <= cutoff]
    require(len(tail) >= 40, f"too few tail observations for {ticker}")
    worst_day = r.idxmin()
    rows.append(
        {
            "ticker": ticker,
            "annualized_volatility": r.std() * np.sqrt(252),
            "var_5": cutoff,
            "expected_shortfall_5": tail.mean(),
            "max_drawdown": max_drawdown(r),
            "worst_day": worst_day.date(),
            "worst_day_return": r.loc[worst_day],
            "skew": r.skew(),
            "kurtosis": r.kurt(),
        }
    )

summary = pd.DataFrame(rows).sort_values("expected_shortfall_5")
require(summary["expected_shortfall_5"].lt(0).all(), "expected shortfall should be negative for all ETFs")
require(summary["annualized_volatility"].between(0, 2).all(), "annualized volatility outside expected range")

apply_dark_theme()
fig, ax = plt.subplots(figsize=(10, 5))
colors = ["#ef4444" if ticker == summary.iloc[0]["ticker"] else "#3b82f6" for ticker in summary["ticker"]]
ax.barh(summary["ticker"], summary["expected_shortfall_5"], color=colors)
ax.set_title("Commodity ETF Tail Risk")
ax.set_xlabel("Average Daily Return in Worst 5% of Days")
ax.set_ylabel("Commodity ETF")
ax.axvline(0, color="#555555", linewidth=0.8)
ax.grid(axis="x", color="#2a2a2a", linewidth=0.6, alpha=0.55)
plt.tight_layout()
CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(CHART_PATH, dpi=150, facecolor=fig.get_facecolor())
plt.close(fig)

print("=== Commodity ETF Tail Risk Screen ===")
print("Universe: GLD, SLV, USO, UNG, DBA, DBB, DBC")
print(f"Sample: {returns.index.min().date()} to {returns.index.max().date()} ({len(returns)} trading days)")
print(f"Tail metric: expected shortfall across the worst {TAIL_PROBABILITY:.0%} of daily returns")
print()
print("Tail-risk ranking:")
for _, row in summary.iterrows():
    print(
        f"{row['ticker']:4s}  vol={fmt_abs_pct(row['annualized_volatility'])}  "
        f"VaR_5={fmt_pct(row['var_5'])}  "
        f"ES_5={fmt_pct(row['expected_shortfall_5'])}  "
        f"max_drawdown={fmt_pct(row['max_drawdown'])}  "
        f"worst_day={row['worst_day']} {fmt_pct(row['worst_day_return'])}  "
        f"skew={row['skew']:+.2f}  kurtosis={row['kurtosis']:+.2f}"
    )
print()
print(f"Highest tail risk: {summary.iloc[0]['ticker']} with ES_5={fmt_pct(summary.iloc[0]['expected_shortfall_5'])}")
print(f"Lowest tail risk:  {summary.iloc[-1]['ticker']} with ES_5={fmt_pct(summary.iloc[-1]['expected_shortfall_5'])}")
