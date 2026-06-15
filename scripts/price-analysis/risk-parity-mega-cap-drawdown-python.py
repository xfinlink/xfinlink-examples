# Full write-up: https://xfinlink.com/blog/risk-parity-mega-cap-drawdown-python

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "risk-parity-mega-cap-drawdown-python"
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOG", "JPM", "XOM", "JNJ", "PG"]
LOOKBACK_DAYS = 63


def fmt_pct(value: float) -> str:
    return f"{value * 100:6.1f}%"


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1 + returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return float(drawdown.min())


def annualized_return(returns: pd.Series) -> float:
    return float((1 + returns).prod() ** (252 / len(returns)) - 1)


def portfolio_stats(returns: pd.Series) -> dict[str, float]:
    ann_vol = float(returns.std() * np.sqrt(252))
    ann_return = annualized_return(returns)
    return {
        "ann_return": ann_return,
        "ann_vol": ann_vol,
        "sharpe": ann_return / ann_vol if ann_vol else np.nan,
        "max_drawdown": max_drawdown(returns),
    }


def make_chart(equal_returns: pd.Series, rp_returns: pd.Series) -> None:
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
    equal_curve = (1 + equal_returns).cumprod() - 1
    rp_curve = (1 + rp_returns).cumprod() - 1
    ax.plot(equal_curve.index, equal_curve * 100, color="#6b7280", linewidth=2, label="Equal weight")
    ax.plot(rp_curve.index, rp_curve * 100, color="#3b82f6", linewidth=2.4, label="Inverse-volatility weight")
    ax.axhline(0, color="#e0e0e0", linewidth=1, alpha=0.35)
    ax.set_title("Risk parity versus equal weight for mega-cap stocks")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative return")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    out = Path("worker/src/site/blog-images") / f"{SLUG}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, facecolor="#0a0a0a")
    plt.close(fig)


prices = xfl.prices(TICKERS, period="3y", fields=["adj_close"], max_rows=50000)
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
if len(price_table) < 500:
    raise ValueError(f"Not enough common price history: {len(price_table)} rows")

returns = price_table.pct_change().dropna()
rolling_vol = returns.rolling(LOOKBACK_DAYS).std() * np.sqrt(252)
inverse_vol = 1 / rolling_vol.shift(1).replace(0, np.nan)
weights = inverse_vol.div(inverse_vol.sum(axis=1), axis=0).dropna()

if weights.empty:
    raise ValueError("Risk-parity weights are empty")
if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-6):
    raise ValueError("Risk-parity weights do not sum to 1")

aligned_returns = returns.loc[weights.index]
risk_parity_returns = (weights * aligned_returns).sum(axis=1)
equal_weight_returns = aligned_returns.mean(axis=1)

if risk_parity_returns.isna().any() or equal_weight_returns.isna().any():
    raise ValueError("Portfolio return series contains NaN values")

equal_stats = portfolio_stats(equal_weight_returns)
rp_stats = portfolio_stats(risk_parity_returns)
latest_weights = weights.iloc[-1].sort_values(ascending=False)

make_chart(equal_weight_returns, risk_parity_returns)

print("=== Mega-Cap Risk-Parity Drawdown Test ===")
print(f"Universe: {len(TICKERS)} mega-cap stocks")
print(f"Sample: {aligned_returns.index.min().date()} to {aligned_returns.index.max().date()} ({len(aligned_returns)} trading days)")
print(f"Volatility lookback: {LOOKBACK_DAYS} trading days")
print()
print("Portfolio comparison:")
print(
    f"Equal weight      return={fmt_pct(equal_stats['ann_return'])}  "
    f"vol={fmt_pct(equal_stats['ann_vol'])}  "
    f"max_drawdown={fmt_pct(equal_stats['max_drawdown'])}  "
    f"Sharpe={equal_stats['sharpe']:5.2f}"
)
print(
    f"Inverse-vol weight return={fmt_pct(rp_stats['ann_return'])}  "
    f"vol={fmt_pct(rp_stats['ann_vol'])}  "
    f"max_drawdown={fmt_pct(rp_stats['max_drawdown'])}  "
    f"Sharpe={rp_stats['sharpe']:5.2f}"
)
print()
print("Latest inverse-volatility weights:")
for ticker, weight in latest_weights.items():
    print(f"{ticker:<5} {fmt_pct(weight)}")
