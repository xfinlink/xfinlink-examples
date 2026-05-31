# Full write-up: https://xfinlink.com/blog/defensive-stocks-ai-drawdown-hedge-python

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xfinlink as xfl

xfl.set_api_key(os.environ.get("XFINLINK_API_KEY", "YOUR_API_KEY"))  # free at https://xfinlink.com/signup

SLUG = "defensive-stocks-ai-drawdown-hedge-python"
CHART_PATH = f"/Users/lyonghee97/Desktop/xfinlink/worker/src/site/blog-images/{SLUG}.png"

AI_TICKERS = ["NVDA", "MSFT", "ORCL", "AMZN", "META"]
DEFENSIVE_TICKERS = ["PG", "JNJ", "KO", "WMT", "MCD"]
MARKET = "SPY"
TICKERS = AI_TICKERS + DEFENSIVE_TICKERS + [MARKET]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


df = xfl.prices(TICKERS, period="5y", fields=["close"])
df = df[df["ticker"].isin(TICKERS)].copy()

require(not df.empty, "prices returned no rows")
require(set(TICKERS).issubset(set(df["ticker"])), "missing one or more requested tickers")

prices = (
    df.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    .sort_index()
    .dropna(subset=TICKERS)
)
require(len(prices) >= 750, "not enough complete price observations")

monthly_prices = prices.resample("ME").last()
monthly_returns = monthly_prices.pct_change().dropna()

monthly_returns["AI basket"] = monthly_returns[AI_TICKERS].mean(axis=1)
monthly_returns["Defensive basket"] = monthly_returns[DEFENSIVE_TICKERS].mean(axis=1)
monthly_returns["SPY"] = monthly_returns[MARKET]

threshold = monthly_returns["AI basket"].quantile(0.10)
stress = monthly_returns[monthly_returns["AI basket"] <= threshold].copy()
normal = monthly_returns[monthly_returns["AI basket"] > threshold].copy()

require(len(stress) >= 5, "not enough stress months for a bottom-decile test")

summary = pd.DataFrame(
    {
        "Stress months": stress[["AI basket", "Defensive basket", "SPY"]].mean(),
        "Other months": normal[["AI basket", "Defensive basket", "SPY"]].mean(),
    }
)

protection_spread = stress["Defensive basket"].mean() - stress["AI basket"].mean()
hit_rate = (stress["Defensive basket"] > stress["AI basket"]).mean()
spy_hit_rate = (stress["Defensive basket"] > stress["SPY"]).mean()
worst = stress.sort_values("AI basket").iloc[0]

print("=== Defensive Stocks During AI Drawdowns ===")
print(f"Sample: {monthly_returns.index.min().strftime('%Y-%m')} to {monthly_returns.index.max().strftime('%Y-%m')}")
print(f"AI stress threshold (bottom decile): {threshold:.1%}")
print(f"Stress months: {len(stress)} of {len(monthly_returns)}")
print()
print("Average monthly returns:")
print(f"AI basket, stress months:        {stress['AI basket'].mean():+.1%}")
print(f"Defensive basket, stress months: {stress['Defensive basket'].mean():+.1%}")
print(f"SPY, stress months:              {stress['SPY'].mean():+.1%}")
print(f"Protection spread:               {protection_spread:+.1%}")
print()
print(f"Defensive basket beat AI in {hit_rate:.0%} of stress months")
print(f"Defensive basket beat SPY in {spy_hit_rate:.0%} of stress months")
print(
    f"Worst AI month: {worst.name.strftime('%Y-%m')}  "
    f"AI={worst['AI basket']:+.1%}  Defensive={worst['Defensive basket']:+.1%}  SPY={worst['SPY']:+.1%}"
)

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

fig, ax = plt.subplots(figsize=(10, 5))
labels = summary.index.tolist()
x = np.arange(len(labels))
width = 0.34

stress_bars = ax.bar(x - width / 2, summary["Stress months"] * 100, width, color="#3b82f6", label="Stress months")
normal_bars = ax.bar(x + width / 2, summary["Other months"] * 100, width, color="#3a3a3a", label="Other months")

ax.axhline(0, color="#e0e0e0", linewidth=0.8)
ax.set_title("Defensive Stocks During AI Drawdowns")
ax.set_xlabel("Basket")
ax.set_ylabel("Average monthly return (%)")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(frameon=False)
ax.grid(axis="y", color="#2a2a2a", linewidth=0.6, alpha=0.55)

for bars in [stress_bars, normal_bars]:
    for bar in bars:
        value = bar.get_height()
        va = "bottom" if value >= 0 else "top"
        offset = 0.3 if value >= 0 else -0.3
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset, f"{value:+.1f}%", ha="center", va=va)

plt.tight_layout()
plt.savefig(CHART_PATH, dpi=150, facecolor=fig.get_facecolor())
