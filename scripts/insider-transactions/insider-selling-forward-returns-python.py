# Full write-up: https://xfinlink.com/blog/insider-selling-forward-returns-python

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "insider-selling-forward-returns-python"
TICKERS = ["TSLA", "NVDA", "AMZN", "COIN", "PLTR", "CRM", "ORCL", "AVGO", "MSFT"]
HORIZON_DAYS = 63


def fmt_pct(value: float) -> str:
    return f"{value * 100:6.1f}%"


def fmt_money_m(value: float) -> str:
    return f"${value / 1_000_000:,.0f}M"


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

    ordered = summary.sort_values("spread")
    colors = ["#3b82f6" if value < 0 else "#6b7280" for value in ordered["spread"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(ordered.index, ordered["spread"] * 100, color=colors)
    ax.axvline(0, color="#e0e0e0", linewidth=1, alpha=0.55)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, color="#e0e0e0", fontsize=8)
    ax.set_title("Forward return after heavy insider-selling months")
    ax.set_xlabel("Heavy-selling return minus other insider-active months")
    ax.set_ylabel("Ticker")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    out = Path("worker/src/site/blog-images") / f"{SLUG}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, facecolor="#0a0a0a")
    plt.close(fig)


insiders = xfl.insiders(
    TICKERS,
    period="3y",
    transaction_type=["open_market_buy", "open_market_sell"],
    fields=["ticker", "transaction_date", "transaction_type", "transaction_value"],
    max_rows=20000,
)
if insiders.empty:
    raise ValueError("Insider transaction DataFrame is empty")

required_insiders = {"ticker", "transaction_date", "transaction_type", "transaction_value"}
missing_insiders = required_insiders - set(insiders.columns)
if missing_insiders:
    raise ValueError(f"Missing insider columns: {sorted(missing_insiders)}")

insiders = insiders.copy()
insiders["transaction_date"] = pd.to_datetime(insiders["transaction_date"]).dt.tz_localize(None)
insiders["month"] = insiders["transaction_date"].dt.to_period("M").dt.to_timestamp("M")
insiders["transaction_value"] = pd.to_numeric(insiders["transaction_value"], errors="coerce").fillna(0.0)
insiders["sell_value"] = np.where(insiders["transaction_type"].eq("open_market_sell"), insiders["transaction_value"], 0.0)
insiders["buy_value"] = np.where(insiders["transaction_type"].eq("open_market_buy"), insiders["transaction_value"], 0.0)

monthly = insiders.groupby(["ticker", "month"], as_index=False)[["sell_value", "buy_value"]].sum()
monthly["net_sell_value"] = monthly["sell_value"] - monthly["buy_value"]
if monthly.empty:
    raise ValueError("Monthly insider flow table is empty")

prices = xfl.prices(TICKERS, period="3y", fields=["adj_close"], max_rows=50000)
if prices.empty:
    raise ValueError("Price DataFrame is empty")

price_table = prices.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
missing_prices = sorted(set(TICKERS) - set(price_table.columns))
if missing_prices:
    raise ValueError(f"Missing price columns: {missing_prices}")
if not price_table.index.is_monotonic_increasing:
    raise ValueError("Price dates are not ordered correctly")

events = []
for ticker, group in monthly.groupby("ticker"):
    series = price_table[ticker].dropna()
    positive_sales = group[group["net_sell_value"] > 0]["net_sell_value"]
    if len(positive_sales) < 4:
        continue
    threshold = positive_sales.quantile(0.75)
    for _, row in group.iterrows():
        start_pos = series.index.searchsorted(row["month"])
        end_pos = start_pos + HORIZON_DAYS
        if end_pos >= len(series):
            continue
        events.append({
            "ticker": ticker,
            "month": row["month"],
            "net_sell_value": row["net_sell_value"],
            "heavy_sell": row["net_sell_value"] >= threshold and row["net_sell_value"] > 0,
            "forward_return": series.iloc[end_pos] / series.iloc[start_pos] - 1,
        })

events = pd.DataFrame(events)
if events.empty:
    raise ValueError("No insider flow events survived the forward-return filter")
if events.isna().any().any():
    raise ValueError("Insider event table contains NaN values")
if not events["heavy_sell"].any():
    raise ValueError("No heavy-selling months were identified")

rows = []
for ticker, group in events.groupby("ticker"):
    heavy = group[group["heavy_sell"]]
    other = group[~group["heavy_sell"]]
    if heavy.empty or other.empty:
        continue
    rows.append({
        "ticker": ticker,
        "total_net_sell": group["net_sell_value"].sum(),
        "heavy_months": int(heavy.shape[0]),
        "heavy_forward": heavy["forward_return"].mean(),
        "other_forward": other["forward_return"].mean(),
        "spread": heavy["forward_return"].mean() - other["forward_return"].mean(),
    })

summary = pd.DataFrame(rows).set_index("ticker")
if summary.empty:
    raise ValueError("Ticker-level insider summary is empty")

make_chart(summary)

heavy_all = events[events["heavy_sell"]]
other_all = events[~events["heavy_sell"]]
overall_spread = heavy_all["forward_return"].mean() - other_all["forward_return"].mean()

print("=== Insider Selling Forward-Return Test ===")
print(f"Transactions analyzed: {len(insiders):,}")
print(f"Event months with forward returns: {len(events)}")
print(f"Heavy-selling definition: top quartile of positive monthly net selling per ticker")
print(f"Heavy-selling months: {len(heavy_all)}, avg forward return {fmt_pct(heavy_all['forward_return'].mean())}")
print(f"Other insider-active months: {len(other_all)}, avg forward return {fmt_pct(other_all['forward_return'].mean())}")
print(f"Heavy-minus-other spread: {fmt_pct(overall_spread)}")
print()
print("Ticker-level event results:")
for ticker, row in summary.sort_values("spread").iterrows():
    print(
        f"{ticker:<5} net_sell={fmt_money_m(row['total_net_sell']):>9}  "
        f"heavy_months={int(row['heavy_months']):2d}  "
        f"heavy_fwd={fmt_pct(row['heavy_forward'])}  "
        f"other_fwd={fmt_pct(row['other_forward'])}  "
        f"spread={fmt_pct(row['spread'])}"
    )
