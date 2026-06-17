# Full write-up: https://xfinlink.com/blog/ai-momentum-leadership-backtest-python

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "ai-momentum-leadership-backtest-python"
CHART_PATH = Path("worker/src/site/blog-images") / f"{SLUG}.png"
TICKERS = ["NVDA", "AVGO", "AMD", "MSFT", "META", "AMZN", "GOOG", "ORCL", "PLTR", "SMCI"]
LOOKBACK_MONTHS = 3
TOP_N = 3


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def annualized_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() ** (12 / len(returns)) - 1)


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


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


prices = xfl.prices(TICKERS, period="3y", fields=["adj_close"])
require(not prices.empty, "prices returned no rows")
daily = prices.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
missing = sorted(set(TICKERS) - set(daily.columns))
require(not missing, f"missing price history for {missing}")
daily = daily[TICKERS].dropna()
require(len(daily) > 600, "expected at least 600 complete daily observations")

monthly_prices = daily.resample("ME").last()
monthly_prices = monthly_prices[monthly_prices.index <= daily.index.max()]
if monthly_prices.index[-1] > daily.index.max():
    monthly_prices = monthly_prices.iloc[:-1]
require(len(monthly_prices) >= 24, "expected at least 24 complete monthly observations")

monthly_returns = monthly_prices.pct_change().dropna()
signal = monthly_prices.pct_change(LOOKBACK_MONTHS).shift(1)
rebalance_dates = signal.dropna().index.intersection(monthly_returns.index)
require(len(rebalance_dates) >= 24, "expected at least 24 rebalance months")

top_returns = []
leaders = []
latest_signal = signal.loc[rebalance_dates[-1]].sort_values(ascending=False)
for dt in rebalance_dates:
    selected = signal.loc[dt].sort_values(ascending=False).head(TOP_N).index.tolist()
    leaders.extend(selected)
    top_returns.append(monthly_returns.loc[dt, selected].mean())

top3 = pd.Series(top_returns, index=rebalance_dates, name="top3")
equal_weight = monthly_returns.loc[rebalance_dates].mean(axis=1)
leader_counts = pd.Series(leaders).value_counts()

comparison = pd.DataFrame({"Top-3 momentum": (1 + top3).cumprod(), "Equal AI basket": (1 + equal_weight).cumprod()})

apply_dark_theme()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(comparison.index, comparison["Top-3 momentum"], color="#3b82f6", linewidth=2.4, label="Top-3 momentum")
ax.plot(comparison.index, comparison["Equal AI basket"], color="#22c55e", linewidth=2.0, label="Equal AI basket")
ax.set_title("AI Momentum Leadership Backtest")
ax.set_xlabel("Month")
ax.set_ylabel("Growth of $1")
ax.legend(frameon=False)
ax.grid(axis="y", color="#2a2a2a", linewidth=0.6, alpha=0.55)
plt.tight_layout()
CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(CHART_PATH, dpi=150, facecolor=fig.get_facecolor())
plt.close(fig)

print("=== AI Momentum Leadership Backtest ===")
print("Universe: 10 AI-linked stocks")
print(f"Sample: {rebalance_dates.min().date()} to {rebalance_dates.max().date()} ({len(rebalance_dates)} monthly rebalances)")
print(f"Signal: prior {LOOKBACK_MONTHS}-month return; portfolio: top {TOP_N} equal-weighted names")
print()
print("Portfolio comparison:")
print(
    f"Top-3 momentum   return={fmt_pct(annualized_return(top3))}  "
    f"vol={top3.std() * np.sqrt(12):5.1%}  "
    f"max_drawdown={max_drawdown(top3):6.1%}  "
    f"positive_months={(top3 > 0).mean():5.1%}"
)
print(
    f"Equal AI basket   return={fmt_pct(annualized_return(equal_weight))}  "
    f"vol={equal_weight.std() * np.sqrt(12):5.1%}  "
    f"max_drawdown={max_drawdown(equal_weight):6.1%}  "
    f"positive_months={(equal_weight > 0).mean():5.1%}"
)
print()
print("Most frequent momentum leaders:")
for ticker, count in leader_counts.items():
    print(f"{ticker:5s} selected in {count:2d} months")
print()
print("Latest signal leaders:")
for ticker, value in latest_signal.head(TOP_N).items():
    print(f"{ticker:5s} trailing_3m_return={fmt_pct(value)}")
