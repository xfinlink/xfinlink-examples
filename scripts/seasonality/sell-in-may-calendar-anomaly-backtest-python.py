# Full write-up: https://xfinlink.com/blog/sell-in-may-calendar-anomaly-backtest-python

import xfinlink as xfl
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

xfl.set_api_key("YOUR_API_KEY")  # free at https://xfinlink.com/signup

tickers = ["SPY", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI"]
df = xfl.prices(tickers, period="10y", fields=["close", "return_daily"])

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values(["ticker", "date"])

results = []
for ticker in tickers:
    t = df[df["ticker"] == ticker].copy()
    t = t.set_index("date").sort_index()
    years = sorted(t.index.year.unique())
    winter_rets, summer_rets = [], []

    for year in years:
        winter = t.loc[f"{year-1}-11-01":f"{year}-04-30", "return_daily"].dropna()
        summer = t.loc[f"{year}-05-01":f"{year}-10-31", "return_daily"].dropna()
        if len(winter) > 50 and len(summer) > 50:
            winter_rets.append((1 + winter).prod() - 1)
            summer_rets.append((1 + summer).prod() - 1)

    w, s = np.array(winter_rets), np.array(summer_rets)
    t_stat, p_val = stats.ttest_rel(w, s)

    results.append({
        "Ticker": ticker,
        "Avg Nov-Apr (%)": round(np.mean(w) * 100, 2),
        "Avg May-Oct (%)": round(np.mean(s) * 100, 2),
        "Diff (pp)": round((np.mean(w) - np.mean(s)) * 100, 2),
        "Win Rate (%)": round(np.mean(w > s) * 100, 1),
        "t-stat": round(t_stat, 2),
        "p-value": round(p_val, 4),
        "n_years": len(winter_rets)
    })

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

sig = results_df[results_df["p-value"] < 0.05]
print(f"\nStatistically significant (p < 0.05): {len(sig)} of {len(results_df)} ETFs")
print(f"Average winter-summer difference across all ETFs: {results_df['Diff (pp)'].mean():.2f} pp")

# ── Chart ──────────────────────────────────────────────────────────────
winter_vals = results_df["Avg Nov-Apr (%)"].values
summer_vals = results_df["Avg May-Oct (%)"].values
x = np.arange(len(tickers))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor("#0a0a0a")
ax.set_facecolor("#0a0a0a")

bars1 = ax.bar(x - width / 2, winter_vals, width, label="Nov-Apr (Winter)", color="#3b82f6")
bars2 = ax.bar(x + width / 2, summer_vals, width, label="May-Oct (Summer)", color="#f59e0b")

ax.set_xlabel("ETF", color="#e0e0e0", fontsize=12)
ax.set_ylabel("Average 6-Month Return (%)", color="#e0e0e0", fontsize=12)
ax.set_title("Sell in May: Winter vs Summer Returns (2016-2025)", color="#e0e0e0", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(tickers, color="#e0e0e0", fontsize=11)
ax.tick_params(axis="y", colors="#e0e0e0")
ax.legend(facecolor="#1a1a1a", edgecolor="#333333", labelcolor="#e0e0e0", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_color("#333333")
ax.spines["left"].set_color("#333333")

for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., h + 0.3,
            f"{h:.1f}%", ha="center", va="bottom", color="#e0e0e0", fontsize=8)
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2., h + 0.3,
            f"{h:.1f}%", ha="center", va="bottom", color="#e0e0e0", fontsize=8)

plt.tight_layout()
plt.savefig("sell-in-may-calendar-anomaly-backtest-python.png", dpi=150, facecolor="#0a0a0a")
print("\nChart saved to sell-in-may-calendar-anomaly-backtest-python.png")
