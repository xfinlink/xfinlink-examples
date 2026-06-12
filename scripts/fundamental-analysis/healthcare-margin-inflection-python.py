# Full write-up: https://xfinlink.com/blog/healthcare-margin-inflection-python

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "healthcare-margin-inflection-python"
TICKERS = ["UNH", "JNJ", "LLY", "MRK", "ABBV", "PFE"]
HORIZON_DAYS = 126


def fmt_pct(value: float) -> str:
    return f"{value * 100:6.1f}%"


def make_chart(events: pd.DataFrame) -> None:
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
    colors = ["#3b82f6" if value > 0 else "#6b7280" for value in events["fcf_margin_delta"]]
    ax.scatter(
        events["fcf_margin_delta"] * 100,
        events["forward_return"] * 100,
        color=colors,
        alpha=0.82,
        s=58,
    )
    ax.axhline(0, color="#e0e0e0", linewidth=1, alpha=0.45)
    ax.axvline(0, color="#e0e0e0", linewidth=1, alpha=0.45)
    ax.set_title("Healthcare cash-flow margin inflections and six-month forward returns")
    ax.set_xlabel("Free-cash-flow margin change versus prior annual filing")
    ax.set_ylabel("Forward 126-trading-day return")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    plt.tight_layout()
    out = Path("worker/src/site/blog-images") / f"{SLUG}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, facecolor="#0a0a0a")
    plt.close(fig)


fund = xfl.fundamentals(
    TICKERS,
    period_type="annual",
    period="10y",
    fields=["revenue", "free_cash_flow"],
    max_rows=5000,
)
if fund.empty:
    raise ValueError("Fundamentals DataFrame is empty")

required_fund = {"ticker", "period_end", "filing_date", "revenue", "free_cash_flow"}
missing_fund = required_fund - set(fund.columns)
if missing_fund:
    raise ValueError(f"Missing fundamentals columns: {sorted(missing_fund)}")

prices = xfl.prices(TICKERS, start="2016-01-01", fields=["adj_close"], max_rows=50000)
if prices.empty:
    raise ValueError("Price DataFrame is empty")

price_table = prices.pivot_table(index="date", columns="ticker", values="adj_close").sort_index()
missing_prices = sorted(set(TICKERS) - set(price_table.columns))
if missing_prices:
    raise ValueError(f"Missing price columns: {missing_prices}")
if not price_table.index.is_monotonic_increasing:
    raise ValueError("Price dates are not ordered correctly")

events = []
for ticker, group in fund.sort_values(["ticker", "period_end"]).groupby("ticker"):
    group = group.dropna(subset=["revenue", "free_cash_flow", "filing_date"]).copy()
    if (group["revenue"] <= 0).any():
        raise ValueError(f"Revenue should be positive for {ticker}")
    group["fcf_margin"] = group["free_cash_flow"] / group["revenue"]
    group = group[group["fcf_margin"].between(-0.5, 0.8)]
    group["fcf_margin_delta"] = group["fcf_margin"].diff()

    series = price_table[ticker].dropna()
    for _, row in group.dropna(subset=["fcf_margin_delta"]).iterrows():
        filing_date = pd.Timestamp(row["filing_date"])
        if filing_date.tzinfo:
            filing_date = filing_date.tz_localize(None)
        start_pos = series.index.searchsorted(filing_date)
        end_pos = start_pos + HORIZON_DAYS
        if end_pos >= len(series):
            continue
        events.append({
            "ticker": ticker,
            "filing_date": filing_date,
            "fcf_margin": row["fcf_margin"],
            "fcf_margin_delta": row["fcf_margin_delta"],
            "forward_return": series.iloc[end_pos] / series.iloc[start_pos] - 1,
        })

events = pd.DataFrame(events)
if events.empty:
    raise ValueError("No margin inflection events survived the forward-return filter")
if events.isna().any().any():
    raise ValueError("Margin event table contains NaN values")

make_chart(events)

positive = events[events["fcf_margin_delta"] > 0]
negative = events[events["fcf_margin_delta"] <= 0]
if positive.empty or negative.empty:
    raise ValueError("Need both positive and negative cash-flow margin inflection groups")

spread = positive["forward_return"].mean() - negative["forward_return"].mean()
correlation = events["fcf_margin_delta"].corr(events["forward_return"])
latest = events.sort_values("filing_date").groupby("ticker").tail(1).sort_values("fcf_margin_delta", ascending=False)

print("=== Healthcare Cash-Flow Margin Inflection Signal ===")
print(f"Sample: {events['filing_date'].min().date()} to {events['filing_date'].max().date()} ({len(events)} filing-date observations)")
print(f"Forward horizon: {HORIZON_DAYS} trading days")
print(f"Positive cash-flow margin inflections: {len(positive)} observations, avg forward return {fmt_pct(positive['forward_return'].mean())}")
print(f"Negative or flat inflections: {len(negative)} observations, avg forward return {fmt_pct(negative['forward_return'].mean())}")
print(f"Positive-minus-negative spread: {fmt_pct(spread)}")
print(f"Correlation: cash-flow margin change vs forward return = {correlation:5.2f}")
print()
print("Latest cash-flow margin inflections by ticker:")
for _, row in latest.iterrows():
    print(
        f"{row['ticker']:<5} filing={row['filing_date'].date()}  "
        f"FCF_margin={fmt_pct(row['fcf_margin'])}  "
        f"margin_change={fmt_pct(row['fcf_margin_delta'])}  "
        f"forward_6m={fmt_pct(row['forward_return'])}"
    )
