# Full write-up: https://xfinlink.com/blog/commodity-momentum-rotation-python

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "commodity-momentum-rotation-python"
CHART_PATH = Path("worker/src/site/blog-images") / f"{SLUG}.png"
TICKERS = ["GLD", "SLV", "USO", "UNG", "DBA", "DBB", "DBC"]
LOOKBACK_MONTHS = 6
TOP_N = 2


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


prices = xfl.prices(TICKERS, period="5y", fields=["adj_close", "return_daily"])
require(not prices.empty, "prices returned no rows")

price_daily = prices.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
return_daily = prices.pivot_table(index="date", columns="ticker", values="return_daily").sort_index()
missing = sorted(set(TICKERS) - set(price_daily.columns))
require(not missing, f"missing price history for {missing}")
price_daily = price_daily[TICKERS].dropna()
return_daily = return_daily[TICKERS].dropna()
require(len(price_daily) > 1000, "expected at least 1000 complete daily observations")
require(price_daily.index.is_monotonic_increasing, "dates are not sorted")

monthly_prices = price_daily.resample("ME").last()
monthly_returns = (1 + return_daily).resample("ME").prod() - 1
if monthly_prices.index[-1] > price_daily.index.max():
    monthly_prices = monthly_prices.iloc[:-1]
    monthly_returns = monthly_returns.iloc[:-1]
require(len(monthly_prices) >= 50, "expected at least 50 complete monthly observations")

signal = monthly_prices.pct_change(LOOKBACK_MONTHS).shift(1)
rebalance_dates = signal.dropna().index.intersection(monthly_returns.dropna().index)
require(len(rebalance_dates) >= 36, "expected at least 36 rebalance months")

strategy_returns = []
leaders = []
for dt in rebalance_dates:
    selected = signal.loc[dt].sort_values(ascending=False).head(TOP_N).index.tolist()
    leaders.extend(selected)
    strategy_returns.append(monthly_returns.loc[dt, selected].mean())

strategy = pd.Series(strategy_returns, index=rebalance_dates, name="Top-2 momentum")
equal_weight = monthly_returns.loc[rebalance_dates, TICKERS].mean(axis=1)
broad_commodity = monthly_returns.loc[rebalance_dates, "DBC"]
latest_signal = signal.loc[rebalance_dates[-1]].sort_values(ascending=False)
leader_counts = pd.Series(leaders).value_counts()

require(strategy.notna().all(), "strategy returns contain missing values")
require(abs(annualized_return(strategy)) < 2, "strategy annualized return is implausibly large")

comparison = pd.DataFrame(
    {
        "Top-2 momentum": (1 + strategy).cumprod(),
        "Equal commodity basket": (1 + equal_weight).cumprod(),
        "DBC": (1 + broad_commodity).cumprod(),
    }
)

apply_dark_theme()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(comparison.index, comparison["Top-2 momentum"], color="#3b82f6", linewidth=2.4, label="Top-2 momentum")
ax.plot(comparison.index, comparison["Equal commodity basket"], color="#22c55e", linewidth=2.0, label="Equal basket")
ax.plot(comparison.index, comparison["DBC"], color="#f59e0b", linewidth=1.9, label="DBC")
ax.set_title("Commodity Momentum Rotation")
ax.set_xlabel("Month")
ax.set_ylabel("Growth of $1")
ax.legend(frameon=False)
ax.grid(axis="y", color="#2a2a2a", linewidth=0.6, alpha=0.55)
plt.tight_layout()
CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(CHART_PATH, dpi=150, facecolor=fig.get_facecolor())
plt.close(fig)

print("=== Commodity Momentum Rotation Backtest ===")
print("Universe: GLD, SLV, USO, UNG, DBA, DBB, DBC")
print(f"Sample: {rebalance_dates.min().date()} to {rebalance_dates.max().date()} ({len(rebalance_dates)} monthly rebalances)")
print(f"Signal: prior {LOOKBACK_MONTHS}-month price return; portfolio: top {TOP_N} equal-weighted ETFs")
print()
print("Portfolio comparison:")
print(
    f"Top-2 momentum          return={fmt_pct(annualized_return(strategy))}  "
    f"vol={strategy.std() * np.sqrt(12):5.1%}  "
    f"max_drawdown={max_drawdown(strategy):6.1%}  "
    f"positive_months={(strategy > 0).mean():5.1%}"
)
print(
    f"Equal commodity basket  return={fmt_pct(annualized_return(equal_weight))}  "
    f"vol={equal_weight.std() * np.sqrt(12):5.1%}  "
    f"max_drawdown={max_drawdown(equal_weight):6.1%}  "
    f"positive_months={(equal_weight > 0).mean():5.1%}"
)
print(
    f"DBC broad commodity     return={fmt_pct(annualized_return(broad_commodity))}  "
    f"vol={broad_commodity.std() * np.sqrt(12):5.1%}  "
    f"max_drawdown={max_drawdown(broad_commodity):6.1%}  "
    f"positive_months={(broad_commodity > 0).mean():5.1%}"
)
print()
print("Most frequent momentum leaders:")
for ticker, count in leader_counts.items():
    print(f"{ticker:4s} selected in {count:2d} months")
print()
print("Latest signal ranking:")
for ticker, value in latest_signal.items():
    print(f"{ticker:4s} trailing_6m_return={fmt_pct(value)}")
