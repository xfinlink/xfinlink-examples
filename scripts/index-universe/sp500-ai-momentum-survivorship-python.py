# Full write-up: https://xfinlink.com/blog/sp500-ai-momentum-survivorship-python

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "sp500-ai-momentum-survivorship-python"
CHART_PATH = f"/Users/lyonghee97/Desktop/xfinlink/worker/src/site/blog-images/{SLUG}.png"

AS_OF = "2023-01-03"
AI_TICKERS = ["NVDA", "AVGO", "MSFT", "AMZN", "META", "GOOG", "ORCL", "AMD", "ANET", "VRT", "SMCI", "TSM", "PLTR"]
BENCHMARK = "SPY"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def add_split_adjusted_close(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, group in df.sort_values(["ticker", "date"]).groupby("ticker"):
        adjusted = group.copy()
        split_ratio = adjusted["split_ratio"].fillna(1.0).replace(0, 1.0)
        future_split_factor = split_ratio.shift(-1, fill_value=1.0).iloc[::-1].cumprod().iloc[::-1]
        adjusted["adj_close"] = adjusted["close"] / future_split_factor
        parts.append(adjusted)
    return pd.concat(parts, ignore_index=True)


constituents = xfl.index("sp500", as_of=AS_OF, limit=700).drop_duplicates("ticker")
require(not constituents.empty, "index constituents returned no rows")
sp500_tickers = set(constituents["ticker"])

point_in_time_members = [ticker for ticker in AI_TICKERS if ticker in sp500_tickers]
non_members = [ticker for ticker in AI_TICKERS if ticker not in sp500_tickers]
require(len(point_in_time_members) >= 5, "not enough point-in-time AI members")

prices_raw = xfl.prices(AI_TICKERS + [BENCHMARK], start=AS_OF, fields=["close", "split_ratio"])
prices_raw = prices_raw[prices_raw["ticker"].isin(AI_TICKERS + [BENCHMARK])].copy()
require(not prices_raw.empty, "prices returned no rows")
require(set(AI_TICKERS + [BENCHMARK]).issubset(set(prices_raw["ticker"])), "missing one or more requested tickers")

prices_raw = add_split_adjusted_close(prices_raw)
prices = prices_raw.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last").sort_index().ffill()
valid = [
    ticker
    for ticker in AI_TICKERS + [BENCHMARK]
    if ticker in prices.columns and pd.notna(prices[ticker].iloc[0]) and pd.notna(prices[ticker].iloc[-1])
]
require(set(AI_TICKERS + [BENCHMARK]).issubset(set(valid)), "one or more tickers lacks a complete start/end price")

cumulative_returns = prices[valid] / prices[valid].iloc[0] - 1
final_returns = cumulative_returns.iloc[-1]

member_return = final_returns[point_in_time_members].mean()
non_member_return = final_returns[non_members].mean()
benchmark_return = final_returns[BENCHMARK]
member_contribution = (final_returns[point_in_time_members] / len(point_in_time_members)).sort_values(ascending=False)
ranked_returns = final_returns[AI_TICKERS].sort_values(ascending=False)

print("=== Point-in-Time AI Momentum Test ===")
print(f"Index date: {AS_OF}")
print(f"Price sample: {prices.index.min().date()} to {prices.index.max().date()} ({len(prices)} trading days)")
print()
print(f"AI tickers already in the S&P 500: {', '.join(point_in_time_members)}")
print(f"AI tickers outside the S&P 500 then: {', '.join(non_members)}")
print()
print(f"Equal-weight return, point-in-time S&P 500 AI members: {member_return:+.0%}")
print(f"Equal-weight return, non-member AI basket:             {non_member_return:+.0%}")
print(f"SPY return over same window:                           {benchmark_return:+.0%}")
print()
print("AI ticker returns since the index date:")
for ticker, value in ranked_returns.items():
    membership = "member" if ticker in point_in_time_members else "non-member"
    print(f"{ticker:5s} {value:+7.0%}  {membership}")
print()
print("Top contributors within the point-in-time member basket:")
for ticker, contribution in member_contribution.head(5).items():
    print(f"{ticker:5s} contribution={contribution:+.0%}")

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

fig, ax = plt.subplots(figsize=(10, 7))
colors = ["#3b82f6" if ticker in point_in_time_members else "#f59e0b" for ticker in ranked_returns.index]
ax.barh(ranked_returns.index[::-1], ranked_returns.iloc[::-1] * 100, color=colors[::-1])
ax.axvline(benchmark_return * 100, color="#e0e0e0", linewidth=1.2, linestyle="--", label="SPY")
ax.set_title("Point-in-Time AI Momentum Test")
ax.set_xlabel("Cumulative return since January 3, 2023 (%)")
ax.set_ylabel("Company")
ax.legend(frameon=False)
ax.grid(axis="x", color="#2a2a2a", linewidth=0.6, alpha=0.55)
plt.tight_layout()
plt.savefig(CHART_PATH, dpi=150, facecolor=fig.get_facecolor())
