# Full write-up: https://xfinlink.com/blog/earnings-yield-bond-selloff-python

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "earnings-yield-bond-selloff-python"
STOCKS = ["AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOG", "JPM", "XOM", "JNJ", "PG", "HD", "COST", "CAT", "BA"]
BOND_PROXY = "TLT"
FIELDS = ["market_cap", "pe_ratio", "earnings_yield"]


def fmt_pct(value: float) -> str:
    return f"{value * 100:6.2f}%"


def make_chart(stats: pd.DataFrame) -> None:
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
    labels = stats["regime"].tolist()
    x = range(len(labels))
    width = 0.34
    ax.bar([i - width / 2 for i in x], stats["cheap"] * 100, width=width, color="#3b82f6", label="High earnings yield")
    ax.bar([i + width / 2 for i in x], stats["expensive"] * 100, width=width, color="#6b7280", label="Low earnings yield")
    ax.axhline(0, color="#e0e0e0", linewidth=1, alpha=0.45)
    ax.set_title("Earnings-yield baskets during bond selloffs and rallies")
    ax.set_xlabel("Bond-market regime")
    ax.set_ylabel("Average weekly stock return")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    out = Path("worker/src/site/blog-images") / f"{SLUG}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, facecolor="#0a0a0a")
    plt.close(fig)


metrics = xfl.metrics(STOCKS, period_type="ttm", fields=FIELDS, max_rows=1000)
if metrics.empty:
    raise ValueError("Metrics DataFrame is empty")

required_metrics = {"ticker", "period_end", *FIELDS}
missing_metrics = required_metrics - set(metrics.columns)
if missing_metrics:
    raise ValueError(f"Missing metrics columns: {sorted(missing_metrics)}")

latest = metrics.sort_values("period_end").groupby("ticker").tail(1).dropna(subset=FIELDS).copy()
latest = latest[(latest["market_cap"] > 10_000) & (latest["pe_ratio"] > 0) & (latest["earnings_yield"] > 0)]
if len(latest) < 10:
    raise ValueError(f"Complete valuation universe is too small: {len(latest)}")
if not latest["earnings_yield"].between(0, 0.50).all():
    raise ValueError("Earnings yield contains implausible values")

valuation_rank = latest.sort_values("earnings_yield", ascending=False)
cheap = valuation_rank.head(5)["ticker"].tolist()
expensive = valuation_rank.tail(5)["ticker"].tolist()

prices = xfl.prices([BOND_PROXY, *cheap, *expensive], period="3y", interval="1w", fields=["adj_close"], max_rows=50000)
if prices.empty:
    raise ValueError("Price DataFrame is empty")

required_price = {"ticker", "date", "adj_close"}
missing_price = required_price - set(prices.columns)
if missing_price:
    raise ValueError(f"Missing price columns: {sorted(missing_price)}")

price_table = prices.pivot_table(index="date", columns="ticker", values="adj_close").sort_index().dropna()
missing_tickers = sorted(set([BOND_PROXY, *cheap, *expensive]) - set(price_table.columns))
if missing_tickers:
    raise ValueError(f"Missing price data for: {missing_tickers}")
if not price_table.index.is_monotonic_increasing:
    raise ValueError("Price dates are not ordered correctly")
if len(price_table) < 120:
    raise ValueError(f"Not enough weekly price history: {len(price_table)} rows")

weekly_returns = price_table.pct_change().dropna()
cheap_returns = weekly_returns[cheap].mean(axis=1)
expensive_returns = weekly_returns[expensive].mean(axis=1)
bond_returns = weekly_returns[BOND_PROXY]

selloff_cutoff = bond_returns.quantile(0.20)
rally_cutoff = bond_returns.quantile(0.80)
selloff_mask = bond_returns <= selloff_cutoff
rally_mask = bond_returns >= rally_cutoff

if selloff_mask.sum() == 0 or rally_mask.sum() == 0:
    raise ValueError("Need both bond selloff and rally regimes")

stats = pd.DataFrame([
    {
        "regime": "All weeks",
        "cheap": cheap_returns.mean(),
        "expensive": expensive_returns.mean(),
        "spread": cheap_returns.mean() - expensive_returns.mean(),
        "weeks": len(weekly_returns),
    },
    {
        "regime": "Bond selloffs",
        "cheap": cheap_returns[selloff_mask].mean(),
        "expensive": expensive_returns[selloff_mask].mean(),
        "spread": cheap_returns[selloff_mask].mean() - expensive_returns[selloff_mask].mean(),
        "weeks": int(selloff_mask.sum()),
    },
    {
        "regime": "Bond rallies",
        "cheap": cheap_returns[rally_mask].mean(),
        "expensive": expensive_returns[rally_mask].mean(),
        "spread": cheap_returns[rally_mask].mean() - expensive_returns[rally_mask].mean(),
        "weeks": int(rally_mask.sum()),
    },
])
if stats.isna().any().any():
    raise ValueError("Valuation regime statistics contain NaN values")

make_chart(stats)

print("=== Earnings Yield During Bond Selloffs ===")
print(f"Valuation date range: {latest['period_end'].min().date()} to {latest['period_end'].max().date()}")
print(f"Weekly return sample: {weekly_returns.index.min().date()} to {weekly_returns.index.max().date()} ({len(weekly_returns)} weeks)")
print(f"Bond proxy: {BOND_PROXY}")
print(f"Bond selloff threshold: weekly TLT return <= {fmt_pct(selloff_cutoff)}")
print(f"Bond rally threshold: weekly TLT return >= {fmt_pct(rally_cutoff)}")
print()
print(f"High earnings-yield basket: {', '.join(cheap)}")
print(f"Low earnings-yield basket: {', '.join(expensive)}")
print()
print("Average weekly stock returns:")
for _, row in stats.iterrows():
    print(
        f"{row['regime']:<13} weeks={int(row['weeks']):3d}  "
        f"high_EY={fmt_pct(row['cheap'])}  "
        f"low_EY={fmt_pct(row['expensive'])}  "
        f"spread={fmt_pct(row['spread'])}"
    )
