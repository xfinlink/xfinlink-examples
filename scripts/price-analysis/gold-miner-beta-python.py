# Full write-up: https://xfinlink.com/blog/gold-miner-beta-python

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "gold-miner-beta-python"
CHART_PATH = Path("worker/src/site/blog-images") / f"{SLUG}.png"
TICKERS = ["GLD", "GDX", "NEM", "AEM", "GOLD"]
MINERS = ["GDX", "NEM", "AEM", "GOLD"]
ROLLING_DAYS = 63


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def fmt_pct(value: float) -> str:
    return f"{value:+6.1%}"


def fmt_abs_pct(value: float) -> str:
    return f"{value:5.1%}"


def beta(y: pd.Series, x: pd.Series) -> float:
    return float(y.cov(x) / x.var())


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

rows = []
rolling_betas = pd.DataFrame(index=returns.index)
for ticker in MINERS:
    full_beta = beta(returns[ticker], returns["GLD"])
    corr = returns[ticker].corr(returns["GLD"])
    annualized_vol = returns[ticker].std() * np.sqrt(252)
    gold_up = returns[returns["GLD"] > 0]
    gold_down = returns[returns["GLD"] < 0]
    up_capture = gold_up[ticker].mean() / gold_up["GLD"].mean()
    down_capture = gold_down[ticker].mean() / gold_down["GLD"].mean()

    cov = returns[ticker].rolling(ROLLING_DAYS).cov(returns["GLD"])
    var = returns["GLD"].rolling(ROLLING_DAYS).var()
    rolling_betas[ticker] = cov / var
    rows.append(
        {
            "ticker": ticker,
            "full_sample_beta": full_beta,
            "latest_rolling_beta": rolling_betas[ticker].dropna().iloc[-1],
            "median_rolling_beta": rolling_betas[ticker].median(),
            "correlation": corr,
            "annualized_volatility": annualized_vol,
            "up_capture": up_capture,
            "down_capture": down_capture,
        }
    )

summary = pd.DataFrame(rows).sort_values("full_sample_beta", ascending=False)
require(summary["full_sample_beta"].between(-1, 5).all(), "beta outside expected range")
require(summary["correlation"].between(-1, 1).all(), "correlation outside expected range")

apply_dark_theme()
fig, ax = plt.subplots(figsize=(10, 5))
colors = {"GDX": "#3b82f6", "NEM": "#22c55e", "AEM": "#f59e0b", "GOLD": "#ef4444"}
for ticker in MINERS:
    ax.plot(rolling_betas.index, rolling_betas[ticker], color=colors[ticker], linewidth=1.8, label=ticker)
ax.axhline(1.0, color="#888888", linewidth=0.9, linestyle="--")
ax.set_title("Gold Miner Rolling Beta to GLD")
ax.set_xlabel("Date")
ax.set_ylabel("63-Day Beta to GLD")
ax.legend(frameon=False)
ax.grid(axis="y", color="#2a2a2a", linewidth=0.6, alpha=0.55)
plt.tight_layout()
CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(CHART_PATH, dpi=150, facecolor=fig.get_facecolor())
plt.close(fig)

print("=== Gold Miner Beta to GLD ===")
print("Universe: GDX, NEM, AEM, GOLD versus GLD")
print(f"Sample: {returns.index.min().date()} to {returns.index.max().date()} ({len(returns)} trading days)")
print(f"Rolling beta window: {ROLLING_DAYS} trading days")
print()
print("Beta ranking:")
for _, row in summary.iterrows():
    print(
        f"{row['ticker']:4s}  full_beta={row['full_sample_beta']:5.2f}  "
        f"latest_beta={row['latest_rolling_beta']:5.2f}  "
        f"median_beta={row['median_rolling_beta']:5.2f}  "
        f"corr={row['correlation']:5.2f}  "
        f"vol={fmt_abs_pct(row['annualized_volatility'])}  "
        f"up_capture={row['up_capture']:5.2f}x  "
        f"down_capture={row['down_capture']:5.2f}x"
    )
print()
print(f"Highest beta: {summary.iloc[0]['ticker']} at {summary.iloc[0]['full_sample_beta']:.2f}x GLD")
print(f"Lowest beta:  {summary.iloc[-1]['ticker']} at {summary.iloc[-1]['full_sample_beta']:.2f}x GLD")
