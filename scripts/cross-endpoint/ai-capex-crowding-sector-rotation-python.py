# Full write-up: https://xfinlink.com/blog/ai-capex-crowding-sector-rotation-python

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "ai-capex-crowding-sector-rotation-python"
CHART_PATH = f"/Users/lyonghee97/Desktop/xfinlink/worker/src/site/blog-images/{SLUG}.png"

AI_TICKERS = ["MSFT", "AMZN", "META", "GOOG", "ORCL", "NVDA", "AVGO"]
SECTOR_GROUPS = {
    "Staples": ["PG", "KO", "WMT"],
    "Healthcare": ["JNJ", "UNH", "MRK"],
    "Utilities": ["NEE", "DUK", "SO"],
    "Energy": ["XOM", "CVX", "COP"],
    "Financials": ["JPM", "BAC", "GS"],
    "Industrials": ["CAT", "HON", "GE"],
}
TICKERS = AI_TICKERS + [ticker for group in SECTOR_GROUPS.values() for ticker in group]
WINDOW = 60


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


prices_raw = xfl.prices(TICKERS, period="3y", fields=["adj_close"])
prices_raw = prices_raw[prices_raw["ticker"].isin(TICKERS)].copy()
require(not prices_raw.empty, "prices returned no rows")
require(set(TICKERS).issubset(set(prices_raw["ticker"])), "missing one or more requested tickers")

prices = prices_raw.pivot_table(index="date", columns="ticker", values="adj_close", aggfunc="last").dropna(subset=TICKERS)
returns = prices[TICKERS].pct_change().dropna()
require(len(returns) >= 500, "not enough complete return observations")

returns["AI basket"] = returns[AI_TICKERS].mean(axis=1)
for sector, tickers in SECTOR_GROUPS.items():
    returns[sector] = returns[tickers].mean(axis=1)

rolling_rows = []
for idx in range(WINDOW - 1, len(returns)):
    sample = returns.iloc[idx - WINDOW + 1 : idx + 1]
    pair_corr = sample[AI_TICKERS].corr().to_numpy()
    mask = np.triu(np.ones(pair_corr.shape, dtype=bool), k=1)
    rolling_rows.append(
        {
            "date": returns.index[idx],
            "ai_vol": sample["AI basket"].std() * np.sqrt(252),
            "ai_pair_corr": pair_corr[mask].mean(),
        }
    )

rolling = pd.DataFrame(rolling_rows).set_index("date")
latest = rolling.iloc[-1]
sector_corr = []
for sector in SECTOR_GROUPS:
    sector_corr.append(
        {
            "sector": sector,
            "full_corr": returns[sector].corr(returns["AI basket"]),
            "corr_60d": returns[sector].iloc[-WINDOW:].corr(returns["AI basket"].iloc[-WINDOW:]),
        }
    )
sector_corr = pd.DataFrame(sector_corr).sort_values("corr_60d")

vol_percentile = (rolling["ai_vol"] <= latest["ai_vol"]).mean()
corr_percentile = (rolling["ai_pair_corr"] <= latest["ai_pair_corr"]).mean()

print("=== AI Capex Crowding and Sector Rotation ===")
print(f"Sample: {returns.index.min().date()} to {returns.index.max().date()} ({len(returns)} trading days)")
print(f"Rolling window: {WINDOW} trading days")
print()
print(f"Latest AI basket volatility: {latest['ai_vol']:.1%}")
print(f"Median AI basket volatility: {rolling['ai_vol'].median():.1%}")
print(f"Volatility percentile: {vol_percentile:.0%}")
print(f"Latest AI pairwise correlation: {latest['ai_pair_corr']:.3f}")
print(f"Median AI pairwise correlation: {rolling['ai_pair_corr'].median():.3f}")
print(f"Correlation percentile: {corr_percentile:.0%}")
print()
print("Sector basket correlation to AI basket:")
for _, row in sector_corr.iterrows():
    print(f"{row['sector']:11s} full_period={row['full_corr']:+.3f}  latest_60d={row['corr_60d']:+.3f}")

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

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), gridspec_kw={"width_ratios": [1.25, 1]})
ax1.plot(rolling.index, rolling["ai_vol"] * 100, color="#3b82f6", linewidth=2, label="AI volatility")
ax1.plot(rolling.index, rolling["ai_pair_corr"] * 100, color="#f59e0b", linewidth=2, label="AI pair correlation")
ax1.set_title("AI Basket Crowding Metrics")
ax1.set_xlabel("Date")
ax1.set_ylabel("Percent")
ax1.legend(frameon=False)
ax1.grid(axis="y", color="#2a2a2a", linewidth=0.6, alpha=0.55)

colors = ["#22c55e" if value < 0 else "#3b82f6" for value in sector_corr["corr_60d"]]
ax2.barh(sector_corr["sector"], sector_corr["corr_60d"], color=colors)
ax2.axvline(0, color="#e0e0e0", linewidth=0.8)
ax2.set_title("Latest 60-Day Sector Correlation")
ax2.set_xlabel("Correlation to AI basket")
plt.tight_layout()
plt.savefig(CHART_PATH, dpi=150, facecolor=fig.get_facecolor())
