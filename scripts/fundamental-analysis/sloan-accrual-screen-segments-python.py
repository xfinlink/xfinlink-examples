# Full write-up: https://xfinlink.com/blog/sloan-accrual-screen-segments-python

import xfinlink as xfl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

xfl.api_key = "YOUR_API_KEY"  # free at https://xfinlink.com/signup

# -- Fetch annual fundamentals with segments -----------------------------------
tickers = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "JPM", "V", "UNH",
    "JNJ", "LLY", "MRK", "ABBV", "PG",
    "HD", "COST", "XOM", "CVX", "BAC",
]

df = xfl.fundamentals(
    tickers, period_type="annual", period="3y",
    fields=["net_income", "operating_cash_flow", "total_assets"],
    include_segments=True,
)

# -- Latest annual period per ticker -------------------------------------------
latest = df.sort_values("period_end").groupby("ticker").tail(1).set_index("ticker")

# -- Compute accrual ratio -----------------------------------------------------
latest["accrual_ratio"] = (
    (latest["net_income"] - latest["operating_cash_flow"]) / latest["total_assets"]
).round(3)

# -- Classify quality ----------------------------------------------------------
def classify_quality(ar):
    if ar > 0.03:
        return "LOW"
    elif ar < -0.05:
        return "HIGH"
    else:
        return "MID"

latest["quality"] = latest["accrual_ratio"].apply(classify_quality)

# -- Sort by accrual ratio (most positive = lowest quality) --------------------
ranked = latest.sort_values("accrual_ratio", ascending=False)

# -- Print results -------------------------------------------------------------
print("=== Sloan Accrual Screen: Earnings Quality ===\n")
print(f"{'Ticker':<8s} {'Accrual Ratio':>14s}  {'Quality':<6s}")
print("-" * 32)
for ticker in ranked.index:
    row = ranked.loc[ticker]
    ar = row["accrual_ratio"]
    quality = row["quality"]
    print(f"{ticker:<8s} {ar:>+14.3f}  [{quality}]")

print(f"\nLowest quality:  {', '.join(ranked.index[:5])}")
print(f"Highest quality: {', '.join(ranked.index[-5:])}")

# -- Chart: horizontal bar chart of accrual ratios -----------------------------
fig, ax = plt.subplots(figsize=(10, 10))

colors = []
for ticker in ranked.index:
    ar = ranked.loc[ticker, "accrual_ratio"]
    if ar > 0.03:
        colors.append("#dc2626")
    elif ar < -0.05:
        colors.append("#16a34a")
    else:
        colors.append("#f59e0b")

ax.barh(range(len(ranked)), ranked["accrual_ratio"], color=colors, edgecolor="white")
ax.set_yticks(range(len(ranked)))
ax.set_yticklabels(ranked.index, fontsize=11)
ax.axvline(0, color="black", linewidth=0.8)
ax.axvline(0.03, color="#dc2626", linewidth=1, linestyle="--", alpha=0.5, label="Low quality threshold")
ax.axvline(-0.05, color="#16a34a", linewidth=1, linestyle="--", alpha=0.5, label="High quality threshold")
ax.set_xlabel("Accrual Ratio = (Net Income - OCF) / Total Assets")
ax.set_title(
    "Sloan Accrual Screen: Earnings Quality for 20 Large-Cap Stocks\n"
    "Red = low quality (earnings > cash flow) | Green = high quality (cash > earnings)",
    fontsize=13,
    fontweight="bold",
)
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3, axis="x")
ax.invert_yaxis()

plt.tight_layout()
plt.savefig("sloan-accrual-screen-segments-python.png", dpi=150, bbox_inches="tight")
plt.show()
