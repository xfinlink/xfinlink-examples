# Full write-up: https://xfinlink.com/blog/commodity-risk-parity-python

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "commodity-risk-parity-python"
CHART_PATH = Path("worker/src/site/blog-images") / f"{SLUG}.png"
TICKERS = ["GLD", "SLV", "USO", "UNG", "DBA", "DBB"]
BENCHMARK = "DBC"
ALL_TICKERS = TICKERS + [BENCHMARK]
VOL_WINDOW = 12


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


prices = xfl.prices(ALL_TICKERS, period="5y", fields=["return_daily"])
require(not prices.empty, "prices returned no rows")
daily_returns = prices.pivot_table(index="date", columns="ticker", values="return_daily").sort_index()
missing = sorted(set(ALL_TICKERS) - set(daily_returns.columns))
require(not missing, f"missing return history for {missing}")
daily_returns = daily_returns[ALL_TICKERS].dropna()
require(len(daily_returns) > 1000, "expected at least 1000 complete daily observations")

monthly_returns = (1 + daily_returns).resample("ME").prod() - 1
if monthly_returns.index[-1] > daily_returns.index.max():
    monthly_returns = monthly_returns.iloc[:-1]
require(len(monthly_returns) >= 50, "expected at least 50 monthly observations")

rolling_vol = monthly_returns[TICKERS].rolling(VOL_WINDOW).std().shift(1)
inverse_vol = 1 / rolling_vol.replace(0, np.nan)
weights = inverse_vol.div(inverse_vol.sum(axis=1), axis=0).dropna()
rebalance_dates = weights.index.intersection(monthly_returns.index)
require(len(rebalance_dates) >= 36, "expected at least 36 risk-parity months")
weights = weights.loc[rebalance_dates]
require(np.allclose(weights.sum(axis=1), 1.0), "weights do not sum to one")

risk_parity = (monthly_returns.loc[rebalance_dates, TICKERS] * weights).sum(axis=1)
equal_weight = monthly_returns.loc[rebalance_dates, TICKERS].mean(axis=1)
dbc = monthly_returns.loc[rebalance_dates, BENCHMARK]
latest_weights = weights.iloc[-1].sort_values(ascending=False)

comparison = pd.DataFrame(
    {
        "Inverse-volatility basket": (1 + risk_parity).cumprod(),
        "Equal-weight basket": (1 + equal_weight).cumprod(),
        "DBC": (1 + dbc).cumprod(),
    }
)

apply_dark_theme()
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(comparison.index, comparison["Inverse-volatility basket"], color="#3b82f6", linewidth=2.4, label="Inverse-volatility basket")
ax.plot(comparison.index, comparison["Equal-weight basket"], color="#22c55e", linewidth=2.0, label="Equal-weight basket")
ax.plot(comparison.index, comparison["DBC"], color="#f59e0b", linewidth=1.9, label="DBC")
ax.set_title("Commodity Risk Parity")
ax.set_xlabel("Month")
ax.set_ylabel("Growth of $1")
ax.legend(frameon=False)
ax.grid(axis="y", color="#2a2a2a", linewidth=0.6, alpha=0.55)
plt.tight_layout()
CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(CHART_PATH, dpi=150, facecolor=fig.get_facecolor())
plt.close(fig)

print("=== Commodity Risk Parity Backtest ===")
print("Universe: GLD, SLV, USO, UNG, DBA, DBB")
print(f"Sample: {rebalance_dates.min().date()} to {rebalance_dates.max().date()} ({len(rebalance_dates)} monthly observations)")
print(f"Weight rule: inverse trailing {VOL_WINDOW}-month volatility, rebalanced monthly")
print()
print("Portfolio comparison:")
for label, series in [
    ("Inverse-vol basket", risk_parity),
    ("Equal-weight basket", equal_weight),
    ("DBC benchmark", dbc),
]:
    vol = series.std() * np.sqrt(12)
    sharpe = annualized_return(series) / vol
    print(
        f"{label:20s} return={fmt_pct(annualized_return(series))}  "
        f"vol={vol:5.1%}  "
        f"sharpe={sharpe:5.2f}  "
        f"max_drawdown={max_drawdown(series):6.1%}  "
        f"positive_months={(series > 0).mean():5.1%}"
    )
print()
print("Latest inverse-volatility weights:")
for ticker, weight in latest_weights.items():
    print(f"{ticker:4s} weight={weight:5.1%}")
