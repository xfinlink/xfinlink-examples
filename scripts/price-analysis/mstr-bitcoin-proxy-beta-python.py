# Full write-up: https://xfinlink.com/blog/mstr-bitcoin-proxy-beta-python

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "mstr-bitcoin-proxy-beta-python"
TICKERS = ["MSTR", "IBIT", "COIN", "MARA", "RIOT"]
START = "2024-01-11"


def max_drawdown(series: pd.Series) -> float:
    wealth = series / series.iloc[0]
    return (wealth / wealth.cummax() - 1).min()


def beta_to(reference: pd.Series, target: pd.Series) -> float:
    return target.cov(reference) / reference.var()


def fmt_pct(value: float) -> str:
    return f"{value * 100:6.1f}%"


def make_chart(cumulative: pd.DataFrame, rolling_beta: pd.Series, stats: pd.DataFrame) -> None:
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

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].plot(cumulative.index, cumulative["MSTR"], color="#3b82f6", linewidth=2.2, label="MSTR")
    axes[0].plot(cumulative.index, cumulative["IBIT"], color="#e0e0e0", linewidth=1.8, label="IBIT")
    axes[0].set_title("Cumulative return")
    axes[0].set_ylabel("Growth of $1")
    axes[0].legend(frameon=False)

    bars = axes[1].bar(stats.index, stats["beta_to_ibit"], color="#3b82f6")
    axes[1].axhline(1.0, color="#e0e0e0", linewidth=1, alpha=0.6)
    axes[1].set_title("Full-sample beta to IBIT")
    axes[1].set_ylabel("Beta")
    axes[1].bar_label(bars, fmt="%.2f", padding=3, color="#e0e0e0", fontsize=8)
    axes[1].tick_params(axis="x", rotation=35)

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    fig.suptitle("Is MSTR a leveraged Bitcoin proxy?", color="#e0e0e0", fontsize=14)
    plt.tight_layout()
    out = Path("worker/src/site/blog-images") / f"{SLUG}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, facecolor="#0a0a0a")
    plt.close(fig)


prices = xfl.prices(TICKERS, start=START, fields=["adj_close"])
if prices.empty:
    raise ValueError("Price DataFrame is empty")

pivot = prices.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
missing = sorted(set(TICKERS) - set(pivot.columns))
if missing:
    raise ValueError(f"Missing price columns: {missing}")

pivot = pivot[TICKERS].dropna()
if pivot.empty:
    raise ValueError("No overlapping price history across the Bitcoin-equity basket")
if not pivot.index.is_monotonic_increasing:
    raise ValueError("Dates are not ordered correctly")

returns = pivot.pct_change(fill_method=None).dropna()
if returns.empty or returns[TICKERS].isna().any().any():
    raise ValueError("Return matrix is empty or contains NaN values")

stats = []
for ticker in TICKERS:
    stats.append({
        "ticker": ticker,
        "total_return": pivot[ticker].iloc[-1] / pivot[ticker].iloc[0] - 1,
        "annualized_volatility": returns[ticker].std() * (252 ** 0.5),
        "max_drawdown": max_drawdown(pivot[ticker]),
        "beta_to_ibit": beta_to(returns["IBIT"], returns[ticker]),
        "correlation_to_ibit": returns[ticker].corr(returns["IBIT"]),
    })

stats_df = pd.DataFrame(stats).set_index("ticker")
rolling_beta = returns["MSTR"].rolling(60).cov(returns["IBIT"]) / returns["IBIT"].rolling(60).var()
latest_rolling_beta = rolling_beta.dropna().iloc[-1]
rolling_corr = returns["MSTR"].rolling(60).corr(returns["IBIT"]).dropna().iloc[-1]

down_days = returns["IBIT"] <= -0.03
up_days = returns["IBIT"] >= 0.03
down_capture = returns.loc[down_days, "MSTR"].mean() / returns.loc[down_days, "IBIT"].mean()
up_capture = returns.loc[up_days, "MSTR"].mean() / returns.loc[up_days, "IBIT"].mean()

cumulative = (1 + returns[["MSTR", "IBIT"]]).cumprod()
make_chart(cumulative, rolling_beta, stats_df)

sample_start = returns.index.min().date()
sample_end = returns.index.max().date()

print("=== MSTR Bitcoin-Proxy Beta Test ===")
print(f"Sample: {sample_start} to {sample_end} ({len(returns)} trading days)")
print(f"MSTR full-sample beta to IBIT:        {stats_df.loc['MSTR', 'beta_to_ibit']:.2f}")
print(f"MSTR latest 60-day beta to IBIT:     {latest_rolling_beta:.2f}")
print(f"MSTR full-sample correlation:        {stats_df.loc['MSTR', 'correlation_to_ibit']:.2f}")
print(f"MSTR latest 60-day correlation:      {rolling_corr:.2f}")
print(f"MSTR annualized volatility:          {fmt_pct(stats_df.loc['MSTR', 'annualized_volatility'])}")
print(f"IBIT annualized volatility:          {fmt_pct(stats_df.loc['IBIT', 'annualized_volatility'])}")
print(f"MSTR max drawdown:                   {fmt_pct(stats_df.loc['MSTR', 'max_drawdown'])}")
print(f"IBIT max drawdown:                   {fmt_pct(stats_df.loc['IBIT', 'max_drawdown'])}")
print(f"MSTR up-capture on IBIT +3% days:    {up_capture:.2f}x")
print(f"MSTR down-capture on IBIT -3% days:  {down_capture:.2f}x")
print()
print("Bitcoin-linked equity ranking:")
for ticker, row in stats_df.sort_values("beta_to_ibit", ascending=False).iterrows():
    print(
        f"{ticker:<5} beta={row['beta_to_ibit']:5.2f}  "
        f"corr={row['correlation_to_ibit']:4.2f}  "
        f"vol={fmt_pct(row['annualized_volatility'])}  "
        f"max_drawdown={fmt_pct(row['max_drawdown'])}"
    )
